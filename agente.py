import operator
import re
from typing import Annotated, Sequence, TypedDict

from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from hf_client import HuggingFaceInferenceEmbeddings, chat_completion
from tool import buscar_pedido, calcular_reembolso


class EstadoSoporte(TypedDict):
    mensajes: Annotated[Sequence[BaseMessage], operator.add]


embeddings = HuggingFaceInferenceEmbeddings()
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

tools = [buscar_pedido, calcular_reembolso]

PEDIDO_RE = re.compile(r"\bPED-\d+\b", re.IGNORECASE)
PORCENTAJE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
IMPORTE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:euros?|eur)", re.IGNORECASE)
NUMERO_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _numero(valor: str) -> float:
    return float(valor.replace(",", "."))


def _ejecutar_tools(mensaje: str) -> str:
    resultados = []

    pedido = PEDIDO_RE.search(mensaje)
    if pedido:
        respuesta = buscar_pedido.invoke({"pedido_id": pedido.group(0)})
        resultados.append(f"buscar_pedido: {respuesta}")

    if "reembolso" in mensaje.lower():
        porcentaje = PORCENTAJE_RE.search(mensaje)
        importe = IMPORTE_RE.search(mensaje)
        total = _numero(importe.group(1)) if importe else None
        porcentaje_valor = _numero(porcentaje.group(1)) if porcentaje else None

        if total is None and porcentaje_valor is not None:
            numeros = [_numero(n) for n in NUMERO_RE.findall(mensaje)]
            total = next((n for n in numeros if n != porcentaje_valor), None)

        if total is not None and porcentaje_valor is not None:
            respuesta = calcular_reembolso.invoke(
                {"total": total, "porcentaje": porcentaje_valor}
            )
            resultados.append(f"calcular_reembolso: {respuesta}")

    return "\n".join(resultados)


def _a_mensajes_hf(mensajes: Sequence[BaseMessage]) -> list[dict[str, str]]:
    mensajes_hf = []
    for mensaje in mensajes[-8:]:
        if isinstance(mensaje, HumanMessage):
            role = "user"
        elif isinstance(mensaje, SystemMessage):
            role = "system"
        else:
            role = "assistant"

        if mensaje.content:
            mensajes_hf.append({"role": role, "content": str(mensaje.content)})
    return mensajes_hf


def nodo_llm(estado: EstadoSoporte) -> dict:
    ultimo_humano = next(
        (m.content for m in reversed(estado["mensajes"]) if isinstance(m, HumanMessage)),
        "",
    )
    docs = retriever.invoke(ultimo_humano)
    contexto = "\n".join(d.page_content for d in docs)
    datos_tools = _ejecutar_tools(ultimo_humano)

    system = SystemMessage(
        content=f"""Eres un asistente de soporte amable y preciso.
Usa los datos de herramientas cuando aparezcan para consultar pedidos y calcular reembolsos.
Responde preguntas sobre políticas usando este contexto:

{contexto}

Datos obtenidos con herramientas en este turno:
{datos_tools or "No se ha usado ninguna herramienta en este turno."}

Si no tienes información, dilo claramente. No inventes datos."""
    )

    mensajes_hf = _a_mensajes_hf([system] + list(estado["mensajes"]))
    respuesta = chat_completion(mensajes_hf)
    return {"mensajes": [AIMessage(content=respuesta)]}


grafo = StateGraph(EstadoSoporte)
grafo.add_node("llm", nodo_llm)
grafo.set_entry_point("llm")
grafo.add_edge("llm", END)

checkpointer = MemorySaver()
agente = grafo.compile(checkpointer=checkpointer)


import streamlit as st

from app.rag import EnterpriseRAG

st.set_page_config(
    page_title="Enterprise Private GPT",
    page_icon="🔒",
    layout="wide",
)

st.title("Enterprise Private GPT")
st.caption(
    "Локальный корпоративный RAG-ассистент "
    "на базе Qwen3 8B, E5 и ChromaDB"
)

@st.cache_resource
def load_rag():
    return EnterpriseRAG()


with st.sidebar:
    st.header("Настройки")

    top_k = st.slider(
        "Количество фрагментов (Top-K)",
        min_value=1,
        max_value=8,
        value=4,
    )

    st.info(
        "LLM: Qwen3 8B\n\n"
        "Embeddings: multilingual-e5-base\n\n"
        "Vector DB: ChromaDB"
    )

    if st.button("Очистить историю"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Источники"):
                for source in message["sources"]:
                    st.write(
                        f"[{source['id']}] "
                        f"{source['source']} — "
                        f"стр. {source['page']}"
                    )
                    st.caption(source["text"])


question = st.chat_input(
    "Задайте вопрос по корпоративным документам"
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Поиск и генерация ответа..."):
            try:
                rag = load_rag()
                result = rag.ask(
                    question=question,
                    top_k=top_k,
                )
            except Exception as exc:
                st.error(f"Ошибка RAG: {exc}")
                st.stop()

        st.markdown(result["answer"])

        with st.expander("Источники и контекст"):
            for source in result["sources"]:
                st.markdown(
                    f"**[{source['id']}] "
                    f"{source['source']}, "
                    f"стр. {source['page']}**"
                )

                st.caption(
                    "Cosine distance: "
                    f"{source['distance']:.4f}"
                )

                st.text(source["text"])

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Retrieval",
            f"{result['retrieval_time']:.2f} с",
        )
        col2.metric(
            "Generation",
            f"{result['generation_time']:.2f} с",
        )
        col3.metric(
            "Total latency",
            f"{result['total_time']:.2f} с",
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

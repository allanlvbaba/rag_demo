import os
import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 加载嵌入模型
embedding_model = HuggingFaceEmbeddings(
    model_name="./bge-base-zh-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 初始化 Chroma 客户端
vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embedding_model,
)

# 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 检索方式，similarity 或 mmr
    search_kwargs={"k": 3},
)


# 重述用户消息
load_dotenv()
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
MODEL_NAME = os.getenv("TONGYI_MODEL", "qwen-flash")  # 默认 qwen-flash：当前账号 qwen-turbo 限流为 0

# 将检索到的文档中的 page_content 取出组合到一起
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompt 模板
prompt = PromptTemplate(
    input_variables=["context", "history", "query"],
    template="""
你是一个专业的中文问答助手，擅长基于提供的资料回答问题。
请仅根据以下背景资料以及历史消息回答问题，如无法找到答案，请直接回答“我不知道”。

背景资料：{context}

历史消息：[{history}]

问题：{query}

回答：""",
)

# 大模型
llm = Tongyi(model=MODEL_NAME, api_key=TONGYI_API_KEY)

# 历史消息
history = []

# 格式化历史消息
def format_history(history):
    # 只保留最近 3 轮对话记录
    max_epoch = 3
    if len(history) > 2 * max_epoch:
        history = history[-2 * max_epoch :]
    return "\n".join([f"{i['role']}：{i['content']}" for i in history])

# ！！！rephrase Prompt 模板
rephrase_prompt = PromptTemplate(
    input_variables=["history", "query"],
    template="""
根据历史消息简要完善用户的问题，使其更加具体。只输出完善后的问题。

历史消息：[{history}]

问题：{query}
""",
)

# ！！！重述链条：根据历史和当前 query 生成更具体问题
rephrase_chain = (
    {
        "history": lambda x: format_history(x["history"]),
        "query": lambda x: x["query"],
    }
    | rephrase_prompt
    | llm
    | StrOutputParser()
    | (lambda x: print(f"===== 重述后的查询: {x}=====") or x)
)

# Prompt 模板
prompt = PromptTemplate(
    input_variables=["context", "history", "query"],
    template="""
你是一个专业的中文问答助手，擅长基于提供的资料回答问题。
请仅根据以下背景资料以及历史消息回答问题，如无法找到答案，请直接回答“我不知道”。

背景资料：{context}

历史消息：[{history}]

问题：{query}

回答：""",
)

# ！！！RAG 链条:  使用重述后的 query进行检索
rag_chain = (
    {
        "context": lambda x: format_docs(
            retriever.invoke(
                rephrase_chain.invoke({"history": x.get("history"), "query": x.get("query")}),
                k=3,
            )
        ),
        "history": lambda x: format_history(x.get("history")),
        "query": lambda x: x.get("query"),
    }
    | prompt
    | (lambda x: print(x.text, end="") or x)
    | llm
    | StrOutputParser()  # 输出解析器，将输出解析为字符串
)

query_list = ["不动产或者动产被人占有怎么办", "那要是被损毁了呢"]
for query in query_list:
    print(f"===== 查询: {query} =====")
    response = rag_chain.invoke({"query": query, "history": history})
    print(response, end="\n\n")
    history.extend(
        [
            {"role": "用户", "content": query},
            {"role": "助手", "content": response},
        ]
    )
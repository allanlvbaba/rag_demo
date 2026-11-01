import asyncio

from retrieve_cache_1 import rephrase_retrieve, get_rag_chain, get_llm

# 存储对话历史
chat_history = []

# 1、初始化 LLM
llm = get_llm()

async def invoke_rag(query,conversation_id,chat_history):

    answer = ""

    input={"query":query,"history":chat_history}

    # 1、执行重述、检索
    retrieve_result= rephrase_retrieve(input,llm)
    # 2、获取RAG链
    rag_chain = get_rag_chain(retrieve_result,llm)
    # 3、异步执行RAG链，流式输出
    async for chunk in rag_chain.astream(input):
        answer += chunk
        yield chunk


    # 4、更新对话历史，添加用户查询和AI回答
    chat_history.append(
        {"role": "user", "content": query, "conversation_id": conversation_id}
    )
    chat_history.append(
        {"role": "ai", "content": answer, "conversation_id": conversation_id}
    )


if __name__ == '__main__':
    async def main():

        query_list = ["不动产或者动产被人占有怎么办", "那要是被损毁了呢"]
        for query in query_list:
            print(f"===== 查询: {query} =====")
            async for chunk in invoke_rag(query,1,chat_history):
                print(chunk, end="", flush=True)
            print() # 每个查询结束换行，避免输出被覆盖

    asyncio.run(main())
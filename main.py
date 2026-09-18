from graph.workflow import graph

config = {
    "configurable": {
        "thread_id": "user-1"
    }
}

while True:
    user_message = input("\n🧑 Enter message: ")
    if user_message.lower() in ["exit", "quit"]:
        break
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        },
        config=config
    )
    print("\n🤖 AI:")
    print(result["messages"][-1].content)
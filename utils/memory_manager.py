from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import messages_to_dict, messages_from_dict
import pickle
import os

MEMORY_DIR = "../conversation_memory/"
os.makedirs(MEMORY_DIR, exist_ok=True)

class MemoryManager:
    def __init__(self, session_id="default", window_size=5):
        self.session_id = session_id
        self.memory_file = os.path.join(MEMORY_DIR, f"{session_id}.pkl")
        self.memory = ConversationBufferWindowMemory(
            k=window_size,
            return_messages=True,
            memory_key="chat_history",
            input_key="question"
        )
        self.load_memory()

    def save_memory(self):
        """More robust memory saving"""
        try:
            memory_dict = messages_to_dict(self.memory.chat_memory.messages)
            temp_file = f"{self.memory_file}.tmp"

            with open(temp_file, 'wb') as f:
                pickle.dump(memory_dict, f)

            # Atomic write operation
            if os.path.exists(self.memory_file):
                os.replace(temp_file, self.memory_file)
            else:
                os.rename(temp_file, self.memory_file)
        except Exception as e:
            print(f"Error saving memory: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def load_memory(self):
        """More resilient memory loading"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    memory_dict = pickle.load(f)

                # Validate loaded data
                if isinstance(memory_dict, list):
                    messages = messages_from_dict(memory_dict)
                    if all(hasattr(msg, 'type') for msg in messages):
                        self.memory.chat_memory.messages = messages
                        return

                print("Invalid memory format, clearing memory")
                self.memory.chat_memory.clear()

            except (pickle.PickleError, EOFError, Exception) as e:
                print(f"Error loading memory: {e}")
                self.memory.chat_memory.clear()

    def add_to_memory(self, user_input, ai_response):
        """Add conversation turn to memory"""
        self.memory.save_context(
            {"question": user_input},
            {"answer": ai_response}
        )
        self.save_memory()

    def get_memory(self):
        """Get current memory context"""
        return self.memory.load_memory_variables({})

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.chat_memory.clear()
        self.save_memory()

# Global dictionary to manage multiple memory sessions
_memory_managers = {}

def get_memory_manager(session_id="default", window_size=5):
    """Get or create a memory manager for a specific session"""
    if session_id not in _memory_managers:
        _memory_managers[session_id] = MemoryManager(session_id, window_size)
    return _memory_managers[session_id]
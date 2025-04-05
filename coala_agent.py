import os
import json
import time
from typing import List, Dict, Any
from openai import OpenAI
from datetime import datetime

# Environment Setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Please set your OpenAI API key as an environment variable:")
    print("For PowerShell: $env:OPENAI_API_KEY = 'your-key-here'")
    print("For CMD: set OPENAI_API_KEY=your-key-here")
    exit(1)

# Directory Setup
MEMORY_DIR = "memory_store"
SEMANTIC_DIR = os.path.join(MEMORY_DIR, "semantic")
PROCEDURAL_DIR = os.path.join(MEMORY_DIR, "procedural")
EPISODIC_DIR = os.path.join(MEMORY_DIR, "episodic")

# Create directories if they don't exist
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(SEMANTIC_DIR, exist_ok=True)
os.makedirs(PROCEDURAL_DIR, exist_ok=True)
os.makedirs(EPISODIC_DIR, exist_ok=True)

# Initialize OpenAI client
client = OpenAI()

class CoALAAgent:
    def __init__(self, name: str = "CoALA"):
        self.name = name
        self.working_memory: List[Dict[str, str]] = []
        self.procedural_memory_file = os.path.join(PROCEDURAL_DIR, "guidelines.txt")
        
        # Initialize memory with some basic knowledge
        self._init_semantic_memory()
        
    def _init_semantic_memory(self):
        """Initialize semantic memory with basic knowledge"""
        initial_knowledge = [
            {
                "content": """Working memory is the immediate cognitive workspace that maintains current context and active information.
                It helps in processing immediate tasks and keeping track of ongoing conversation state.""",
                "source": "memory_systems",
                "category": "cognitive_architecture",
                "timestamp": datetime.now().isoformat()
            },
            {
                "content": """Episodic memory stores past experiences and their associated learnings. It allows the system to recall
                similar situations and apply past learnings to new situations.""",
                "source": "memory_systems",
                "category": "cognitive_architecture",
                "timestamp": datetime.now().isoformat()
            },
            {
                "content": """Semantic memory contains factual knowledge and understanding about the world. It provides grounding
                and context for decision making and responses.""",
                "source": "memory_systems",
                "category": "cognitive_architecture",
                "timestamp": datetime.now().isoformat()
            },
            {
                "content": """Procedural memory defines how the system behaves and processes information. It includes both learned
                patterns and explicit rules for interaction.""",
                "source": "memory_systems",
                "category": "cognitive_architecture",
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        for knowledge in initial_knowledge:
            knowledge_file = os.path.join(SEMANTIC_DIR, f"{knowledge['category']}_{int(time.time())}.json")
            with open(knowledge_file, "w") as f:
                json.dump(knowledge, f, indent=2)

    def _get_assistant_response(self, messages: List[Dict[str, str]]) -> str:
        """Get response from OpenAI API"""
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def _reflect_on_conversation(self, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate reflection on conversation using OpenAI"""
        prompt = {
            "role": "system",
            "content": """Analyze this conversation and create a memory reflection following these rules:
            1. For any field where you don't have enough information or the field isn't relevant, use "N/A"
            2. Be extremely concise - each string should be one clear, actionable sentence
            3. Focus only on information that would be useful for handling similar future conversations
            4. Context_tags should be specific enough to match similar situations but general enough to be reusable
            
            Output in this exact format:
            {
                "context_tags": ["tag1", "tag2"],
                "conversation_summary": "One sentence summary",
                "what_worked": "Most effective approach",
                "what_to_avoid": "Most important pitfall"
            }"""
        }
        
        reflection_response = self._get_assistant_response([
            prompt,
            {"role": "user", "content": f"Analyze this conversation: {json.dumps(conversation)}"}
        ])
        
        try:
            return json.loads(reflection_response)
        except:
            return {
                "context_tags": ["error"],
                "conversation_summary": "Failed to parse reflection",
                "what_worked": "N/A",
                "what_to_avoid": "N/A"
            }

    def _update_procedural_memory(self, what_worked: str, what_to_avoid: str):
        """Update procedural memory with new learnings"""
        try:
            with open(self.procedural_memory_file, "r") as f:
                current_rules = f.read()
        except:
            current_rules = ""

        prompt = {
            "role": "system",
            "content": f"""You are maintaining procedural behavior instructions.
            Current rules:
            {current_rules}
            
            New feedback:
            What worked: {what_worked}
            What to avoid: {what_to_avoid}
            
            Generate up to 10 updated, specific, and actionable guidelines.
            Format: 1. [Instruction] - [Brief rationale]"""
        }

        new_rules = self._get_assistant_response([prompt])
        
        with open(self.procedural_memory_file, "w") as f:
            f.write(new_rules)

    def _retrieve_episodic_memory(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant episodic memory from files"""
        memories = []
        for filename in os.listdir(EPISODIC_DIR):
            if filename.endswith('.json'):
                with open(os.path.join(EPISODIC_DIR, filename), 'r') as f:
                    memories.append(json.load(f))
        
        if not memories:
            return None
            
        # Use OpenAI to find the most relevant memory
        prompt = {
            "role": "system",
            "content": f"""Given this query: "{query}"
            And these memories: {json.dumps(memories)}
            Return the index (0-based) of the most relevant memory for this query.
            Return only the number, nothing else."""
        }
        
        try:
            index = int(self._get_assistant_response([prompt]))
            return memories[index]
        except:
            return memories[-1] if memories else None

    def _retrieve_semantic_memory(self, query: str) -> str:
        """Retrieve relevant semantic memory from files"""
        memories = []
        for filename in os.listdir(SEMANTIC_DIR):
            if filename.endswith('.json'):
                with open(os.path.join(SEMANTIC_DIR, filename), 'r') as f:
                    memory = json.load(f)
                    memories.append(memory['content'])
        
        if not memories:
            return ""
            
        # Use OpenAI to find relevant memories
        prompt = {
            "role": "system",
            "content": f"""Given this query: "{query}"
            And these pieces of knowledge: {json.dumps(memories)}
            Return the most relevant pieces of knowledge.
            Return only the knowledge text, nothing else."""
        }
        
        return self._get_assistant_response([prompt])

    def _format_message_history(self) -> str:
        """Format working memory into readable conversation"""
        return "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in self.working_memory
        ])

    def process_message(self, user_message: str) -> str:
        """Process user message and generate response using all memory systems"""
        
        # Add user message to working memory
        self.working_memory.append({"role": "user", "content": user_message})
        
        # Retrieve relevant memories
        episodic_memory = self._retrieve_episodic_memory(user_message)
        semantic_memory = self._retrieve_semantic_memory(user_message)
        
        try:
            with open(self.procedural_memory_file, "r") as f:
                procedural_memory = f.read()
        except:
            procedural_memory = ""
        
        # Construct context-aware prompt
        system_prompt = {
            "role": "system",
            "content": f"""You are {self.name}, an AI assistant with multiple memory systems.
            
            Current conversation context:
            {self._format_message_history()}
            
            Relevant past experience:
            {json.dumps(episodic_memory) if episodic_memory else "No relevant past experiences"}
            
            Relevant knowledge:
            {semantic_memory if semantic_memory else "No relevant knowledge found"}
            
            Behavioral guidelines:
            {procedural_memory if procedural_memory else "No behavioral guidelines established"}
            
            Respond naturally and helpfully to the user's message, incorporating relevant context from your memories."""
        }
        
        # Get response from OpenAI
        response = self._get_assistant_response([system_prompt])
        
        # Add response to working memory
        self.working_memory.append({"role": "assistant", "content": response})
        
        return response
    
    def end_conversation(self):
        """End conversation and update memories"""
        if len(self.working_memory) > 1:  # Only process if there was actual conversation
            # Generate reflection
            reflection = self._reflect_on_conversation(self.working_memory)
            
            # Add timestamp
            reflection["timestamp"] = datetime.now().isoformat()
            
            # Save conversation and reflection
            conversation_file = os.path.join(EPISODIC_DIR, f"conversation_{int(time.time())}.json")
            with open(conversation_file, "w") as f:
                json.dump({
                    "messages": self.working_memory,
                    "reflection": reflection
                }, f, indent=2)
            
            # Update procedural memory
            self._update_procedural_memory(reflection["what_worked"], reflection["what_to_avoid"])
            
            # Clear working memory
            self.working_memory = []

def main():
    """Main interaction loop"""
    print("Initializing CoALA Agent...")
    print(f"Memory storage location: {os.path.abspath(MEMORY_DIR)}")
    
    agent = CoALAAgent()
    print("\nCoALA Agent ready! Type 'exit' to end conversation.")
    print("Type 'status' to see memory statistics.")
    print("Type 'clear' to reset all memories (use with caution).\n")
    
    try:
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'exit':
                print("\nEnding conversation and updating memories...")
                agent.end_conversation()
                print("Goodbye!")
                break
                
            elif user_input.lower() == 'status':
                # Count files in memory directories
                semantic_files = len([f for f in os.listdir(SEMANTIC_DIR) if f.endswith('.json')])
                episodic_files = len([f for f in os.listdir(EPISODIC_DIR) if f.endswith('.json')])
                
                print("\nMemory Status:")
                print(f"Semantic memories: {semantic_files} files")
                print(f"Episodic memories: {episodic_files} files")
                if os.path.exists(agent.procedural_memory_file):
                    with open(agent.procedural_memory_file, 'r') as f:
                        guidelines = len(f.readlines())
                    print(f"Procedural guidelines: {guidelines} rules")
                continue
                
            elif user_input.lower() == 'clear':
                confirm = input("\nWARNING: This will delete all memory files. Type 'yes' to confirm: ")
                if confirm.lower() == 'yes':
                    for f in os.listdir(SEMANTIC_DIR):
                        os.remove(os.path.join(SEMANTIC_DIR, f))
                    for f in os.listdir(EPISODIC_DIR):
                        os.remove(os.path.join(EPISODIC_DIR, f))
                    if os.path.exists(agent.procedural_memory_file):
                        os.remove(agent.procedural_memory_file)
                    print("Memories cleared.")
                continue
            
            response = agent.process_message(user_input)
            print(f"\n{agent.name}: {response}\n")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Ending conversation and updating memories...")
        agent.end_conversation()
        print("Goodbye!")

if __name__ == "__main__":
    main() 
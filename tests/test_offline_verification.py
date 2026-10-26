"""
Rigorous Tests for Verified Offline Execution.
Installs an in-process socket-level connection interceptor (patching socket.socket.connect)
to actively prove that no outbound network calls are made during local embeddings,
ChromaDB, and mock LLM operations.
"""

import os
import socket
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure UTF-8 for console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure no API keys are present
for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY"]:
    os.environ.pop(key, None)

# Force HuggingFace offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --- Install In-Process Socket-Level Connection Interceptor ---
_original_socket_connect = socket.socket.connect
outbound_network_attempts = []


def guarded_socket_connect(self, address):
    """Intercept all socket connection attempts and block any external internet traffic."""
    host = address[0] if isinstance(address, tuple) else address
    outbound_network_attempts.append(str(address))
    # Raise hard exception on any non-local network access
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError(f"VIOLATION: Unintended external network connection attempted to {address} during offline run!")
    return _original_socket_connect(self, address)


# Patch socket globally for the test duration
socket.socket.connect = guarded_socket_connect

import chromadb
from sentence_transformers import SentenceTransformer
from nykaa_support.mock_llm.standalone import GroundedGenerator


def test_socket_firewall_is_active():
    """Verify that our socket interceptor indeed catches and blocks external network requests."""
    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocked = False
    try:
        # Attempt connection to public DNS
        test_socket.connect(("8.8.8.8", 53))
    except RuntimeError as e:
        blocked = True
        print(f"[Socket Interceptor Verified Active]: {e}")
    finally:
        test_socket.close()

    assert blocked is True, "Socket interceptor failed to intercept outbound request"
    outbound_network_attempts.clear()


def test_offline_embeddings_and_chromadb_with_socket_firewall():
    """Run local embeddings, ChromaDB index, and query under active in-process socket interceptor."""
    print("\n[Executing ChromaDB + SentenceTransformer with Active Socket Interceptor]...")

    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

    query = "How many days do I have to return an item?"
    documents = [
        "Nykaa allows returns or replacements within 15 days of delivery for eligible unused products.",
        "Free standard delivery applies to all Nykaa orders over INR 499.",
        "Orders can be cancelled before they leave our warehouse facility."
    ]

    doc_embeddings = model.encode(documents).tolist()
    query_embedding = model.encode([query]).tolist()

    client = chromadb.EphemeralClient()
    collection = client.create_collection("strictly_offline_nykaa_faq")

    collection.add(
        ids=["doc_1", "doc_2", "doc_3"],
        documents=documents,
        embeddings=doc_embeddings,
        metadatas=[{"category": "returns"}, {"category": "shipping"}, {"category": "cancellation"}]
    )

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1
    )

    top_doc = results["documents"][0][0]
    print(f"[Offline ChromaDB Query]: '{query}'")
    print(f"[Offline ChromaDB Top Match]: '{top_doc}'")

    assert "15 days" in top_doc, f"Expected return policy match, got {top_doc}"
    # Verify zero external network attempts were made
    assert len(outbound_network_attempts) == 0, f"External network calls detected: {outbound_network_attempts}"
    print("test_offline_embeddings_and_chromadb_with_socket_firewall: PASSED (Zero outbound network calls)")
    return top_doc


def test_offline_grounded_synthesis_with_socket_firewall(retrieved_context: str):
    """Run deterministic grounded generation under active in-process socket interceptor."""
    generator = GroundedGenerator()
    customer_query = "Can I return a lipstick after 10 days?"
    answer = generator.generate(query=customer_query, context=retrieved_context)

    print("\n[Offline Grounded Response]:\n", answer)
    assert "15 days" in answer
    assert "Nykaa" in answer
    assert len(outbound_network_attempts) == 0, f"External network calls detected: {outbound_network_attempts}"
    print("test_offline_grounded_synthesis_with_socket_firewall: PASSED (Zero outbound network calls)")


if __name__ == "__main__":
    test_socket_firewall_is_active()
    top_doc = test_offline_embeddings_and_chromadb_with_socket_firewall()
    test_offline_grounded_synthesis_with_socket_firewall(top_doc)
    print("\nVerified 100% offline execution with in-process socket-level network blocking verified successfully!")

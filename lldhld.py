from diagrams import Diagram, Cluster, Edge, Custom # Corrected import
from diagrams.programming.framework import React
from diagrams.onprem.compute import Server
from diagrams.onprem.database import Mongodb
from diagrams.onprem.mlops import Cloudpickle 
from diagrams.generic.storage import Storage

# Set attributes for a professional look
graph_attr = {
    "fontsize": "20",
    "bgcolor": "white"
}

with Diagram("Multimodal Math Mentor Architecture", show=False, direction="LR", graph_attr=graph_attr):
    
    # Define User Interface
    user_ui = React("Web/Mobile App\n(Streamlit/React)")

    with Cluster("Backend Services (FastAPI)"):
        api_gateway = Server("API Entry\n(app.py)")
        
        with Cluster("Processing Logic"):
            # Using 'Server' or 'Cloudpickle' as placeholders for logic
            ocr_engine = Server("OCR Processor\n(Tesseract/Vision)")
            solver_engine = Cloudpickle("Reasoning Engine\n(LLM Solver)")
            rag_logic = Server("RAG & Memory\n(rag.py)")

    with Cluster("Data & Persistence Layer"):
        # If you don't have a local path for a custom icon, use a generic Node
        vector_db = Server("Vector DB\n(Chroma/Pinecone)")
        metadata_db = Mongodb("User History\n(MongoDB)")
        blob_store = Storage("Image/Audio\nStorage")

    # Defining the interactions (Data Flow)
    user_ui >> Edge(label="1. Upload Input", color="darkblue") >> api_gateway
    
    api_gateway >> Edge(label="2. Process Image") >> ocr_engine
    ocr_engine >> Edge(label="3. Clean LaTeX") >> solver_engine
    
    solver_engine >> Edge(label="4. Search Context") >> rag_logic
    rag_logic >> vector_db
    
    solver_engine >> Edge(label="5. Log Interaction") >> metadata_db
    
    solver_engine >> Edge(label="6. Return Steps", color="darkgreen") >> api_gateway
    api_gateway >> user_ui
"""
WebSocket handlers for bidirectional communication
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Any
import json
from utils.predictor import predict_complete


async def websocket_handler(websocket: WebSocket):
    """
    Handle WebSocket connections for real-time predictions.
    
    Message format from client:
    {
        "type": "predict",
        "data": { ... patient data ... }
    }
    
    Response format:
    {
        "type": "result",
        "status": "success",
        "data": { ... prediction results ... }
    }
    """
    await websocket.accept()
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "message": "WebSocket connection established"
        })
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "predict":
                    # Get patient data
                    patient_data = message.get("data", {})
                    
                    if not patient_data:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No patient data provided"
                        })
                        continue
                    
                    # Get prediction
                    result = predict_complete(patient_data)
                    
                    # Send result
                    await websocket.send_json({
                        "type": "result",
                        "status": "success",
                        "data": result,
                        "patient_id": patient_data.get("patientId")
                    })
                    
                elif message_type == "ping":
                    # Heartbeat/ping
                    await websocket.send_json({
                        "type": "pong",
                        "message": "Connection alive"
                    })
                    
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    })
                    
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Prediction error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}"
            })
        except:
            pass  # Connection may already be closed

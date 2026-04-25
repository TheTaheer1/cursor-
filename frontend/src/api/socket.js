// WebSocket connection to the FastAPI backend.
// Provides connect(), disconnect(), onMessage(callback), and sendAction(action).

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/live";

let socket = null;
let messageHandler = null;

export function connect(onMessage) {
  messageHandler = onMessage;
  try {
    socket = new WebSocket(WS_URL);
  } catch (e) {
    console.error("[WS] could not open socket", e);
    return;
  }
  socket.onopen = () => console.log("[WS] connected");
  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (messageHandler) messageHandler(data);
    } catch (e) {
      console.error("[WS] parse error", e);
    }
  };
  socket.onerror = (e) => console.error("[WS] error", e);
  socket.onclose = () => console.log("[WS] disconnected");
}

export function disconnect() {
  if (socket) {
    try { socket.close(); } catch (e) { /* noop */ }
    socket = null;
  }
}

export function sendAction(action) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(action));
  }
}

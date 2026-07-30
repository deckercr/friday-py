const SERVER_URL = "ws://localhost:8000/ws/session";
const SAMPLE_RATE = 16000;

const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const responseEl = document.getElementById("response");
const talkButton = document.getElementById("talk-button");

let socket = null;
let audioContext = null;
let scriptNode = null;
let micStream = null;
let recordedSamples = [];
let playbackContext = null;

function floatTo16BitPCM(float32Array) {
  const output = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const clamped = Math.max(-1, Math.min(1, float32Array[i]));
    output[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
  }
  return output;
}

function connect() {
  socket = new WebSocket(SERVER_URL);
  socket.binaryType = "arraybuffer";

  socket.onopen = () => {
    statusEl.textContent = "connected";
  };

  socket.onclose = () => {
    statusEl.textContent = "disconnected, retrying...";
    setTimeout(connect, 2000);
  };

  socket.onmessage = (event) => {
    if (typeof event.data === "string") {
      handleServerMessage(JSON.parse(event.data));
    } else {
      playAudioChunk(event.data);
    }
  };
}

function handleServerMessage(message) {
  if (message.type === "transcript") {
    transcriptEl.textContent = message.text;
  } else if (message.type === "response_text") {
    responseEl.textContent = message.text;
  } else if (message.type === "error") {
    responseEl.textContent = `Error: ${message.message}`;
  }
}

function playAudioChunk(arrayBuffer) {
  if (playbackContext === null) {
    playbackContext = new AudioContext({ sampleRate: SAMPLE_RATE });
  }
  const int16 = new Int16Array(arrayBuffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 0x8000;
  }
  const buffer = playbackContext.createBuffer(1, float32.length, SAMPLE_RATE);
  buffer.copyToChannel(float32, 0);
  const source = playbackContext.createBufferSource();
  source.buffer = buffer;
  source.connect(playbackContext.destination);
  source.start();
}

async function startRecording() {
  recordedSamples = [];
  micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  // Requesting a 16kHz context directly avoids hand-rolled resampling;
  // modern Chromium/Firefox honor this, but it is not guaranteed everywhere.
  audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
  const source = audioContext.createMediaStreamSource(micStream);
  scriptNode = audioContext.createScriptProcessor(4096, 1, 1);
  scriptNode.onaudioprocess = (event) => {
    const channelData = event.inputBuffer.getChannelData(0);
    recordedSamples.push(floatTo16BitPCM(channelData));
  };
  source.connect(scriptNode);
  scriptNode.connect(audioContext.destination);
}

function stopRecording() {
  scriptNode.disconnect();
  audioContext.close();
  micStream.getTracks().forEach((track) => track.stop());

  const totalLength = recordedSamples.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Int16Array(totalLength);
  let offset = 0;
  for (const chunk of recordedSamples) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  socket.send(merged.buffer);
  socket.send(JSON.stringify({ type: "end_utterance" }));
}

talkButton.addEventListener("mousedown", startRecording);
talkButton.addEventListener("mouseup", stopRecording);

connect();

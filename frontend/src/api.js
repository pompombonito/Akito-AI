const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

export async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const detail = typeof payload === "object" ? payload.detail : payload;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return payload;
}

export async function postJson(path, payload) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function streamChat(payload, onSources, onToken, onTask, signal) {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Chat failed (${response.status})`);
  }
  if (!response.body) throw new Error("The server did not return a response stream.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;
  while (!finished) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const eventBlock = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const eventName = eventBlock.match(/^event:\s*(.+)$/m)?.[1]?.trim() || "message";
      const data = eventBlock.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trim()).join("\n");
      if (data) {
        const parsed = JSON.parse(data);
        if (eventName === "sources") onSources(parsed);
        if (eventName === "token") onToken(parsed.text || "");
        if (eventName === "task") onTask?.(parsed);
        if (eventName === "error") throw new Error(parsed.detail || "The model returned an error.");
      }
      boundary = buffer.indexOf("\n\n");
    }
    finished = done;
  }
  if (buffer.trim()) {
    const eventName = buffer.match(/^event:\s*(.+)$/m)?.[1]?.trim();
    const data = buffer.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trim()).join("\n");
    if (eventName === "error" && data) throw new Error(JSON.parse(data).detail || "The model returned an error.");
  }
}

export function fileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(new Error("Could not read the selected image."));
    reader.readAsDataURL(file);
  });
}

export async function recordingToWav(recording, targetRate = 16000) {
  const AudioContextImpl = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextImpl) throw new Error("This browser cannot decode microphone audio. Try a current version of Chrome or Edge.");
  const audioContext = new AudioContextImpl();
  try {
    const decoded = await audioContext.decodeAudioData(await recording.arrayBuffer());
    const sampleCount = Math.max(1, Math.ceil(decoded.duration * targetRate));
    const mono = new Float32Array(sampleCount);
    const channelCount = Math.max(1, decoded.numberOfChannels);
    const sourceStep = decoded.sampleRate / targetRate;
    for (let channel = 0; channel < channelCount; channel += 1) {
      const source = decoded.getChannelData(channel);
      for (let index = 0; index < sampleCount; index += 1) {
        const position = Math.min(index * sourceStep, source.length - 1);
        const lower = Math.floor(position);
        const upper = Math.min(lower + 1, source.length - 1);
        const fraction = position - lower;
        mono[index] += (source[lower] * (1 - fraction) + source[upper] * fraction) / channelCount;
      }
    }

    const bytesPerSample = 2;
    const dataBytes = mono.length * bytesPerSample;
    const wav = new ArrayBuffer(44 + dataBytes);
    const view = new DataView(wav);
    const writeText = (offset, value) => [...value].forEach((character, index) => view.setUint8(offset + index, character.charCodeAt(0)));
    writeText(0, "RIFF");
    view.setUint32(4, 36 + dataBytes, true);
    writeText(8, "WAVE");
    writeText(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, targetRate, true);
    view.setUint32(28, targetRate * bytesPerSample, true);
    view.setUint16(32, bytesPerSample, true);
    view.setUint16(34, 16, true);
    writeText(36, "data");
    view.setUint32(40, dataBytes, true);
    for (let index = 0; index < mono.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, mono[index]));
      view.setInt16(44 + index * bytesPerSample, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    }
    return new Blob([wav], { type: "audio/wav" });
  } finally {
    await audioContext.close().catch(() => {});
  }
}

export async function transcribeRecording(recording, signal) {
  const wav = await recordingToWav(recording);
  const form = new FormData();
  form.append("file", wav, "recording.wav");
  return api("/api/speech/transcribe", { method: "POST", body: form, signal });
}

export function downloadFile(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function downloadSvgAsPng(filename, svg, scale = 2) {
  return new Promise((resolve, reject) => {
    const svgUrl = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml;charset=utf-8" }));
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      const context = canvas.getContext("2d");
      if (!context) {
        URL.revokeObjectURL(svgUrl);
        reject(new Error("This browser could not create the PNG image."));
        return;
      }
      context.fillStyle = "#fffdfa";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((png) => {
        URL.revokeObjectURL(svgUrl);
        if (!png) {
          reject(new Error("Could not export the labeled diagram as PNG."));
          return;
        }
        const pngUrl = URL.createObjectURL(png);
        const link = document.createElement("a");
        link.href = pngUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(pngUrl), 1000);
        resolve();
      }, "image/png");
    };
    image.onerror = () => {
      URL.revokeObjectURL(svgUrl);
      reject(new Error("Could not render the labeled diagram."));
    };
    image.src = svgUrl;
  });
}

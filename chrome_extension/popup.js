const MODAL_ENDPOINT = "https://shahidhkhan--auto-pricer-service-pricingendpoint-web.modal.run/estimate";

document.getElementById("checkBtn").addEventListener("click", async () => {
  const logDiv = document.getElementById("log");
  const resultDiv = document.getElementById("result");
  const errorDiv = document.getElementById("error");
  const btn = document.getElementById("checkBtn");

  logDiv.innerHTML = "";
  resultDiv.textContent = "";
  errorDiv.textContent = "";
  btn.disabled = true;
  btn.textContent = "Checking...";

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    const [{ result: pageText }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => document.body.innerText,
    });

    if (!pageText || pageText.trim().length < 50) {
      throw new Error("Couldn't read enough page text — try a listing page, not a search results page.");
    }

    const response = await fetch(MODAL_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: pageText }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Server error (${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      console.log(`[${(performance.now() / 1000).toFixed(1)}s] chunk received, done=${done}, bytes=${value?.length ?? 0}`); // TEMP DEBUG
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        handleEvent(event);
      }
    }

  } catch (err) {
    errorDiv.textContent = err.message || "Something went wrong.";
  } finally {
    btn.disabled = false;
    btn.textContent = "Check This Listing";
  }
});

function logLine(text) {
  const logDiv = document.getElementById("log");
  const entry = document.createElement("div");
  entry.className = "log-entry";
  entry.textContent = text;
  logDiv.appendChild(entry);
  logDiv.scrollTop = logDiv.scrollHeight;
}

function handleEvent(event) {
  switch (event.stage) {
    case "extracting":
      logLine("🔍 Reading listing...");
      break;
    case "extracted":
      logLine(`✓ Found: ${event.year} ${event.make} ${event.model}`);
      break;
    case "retrieving_comps":
      logLine("📊 Searching for comparable listings...");
      break;
    case "comps_retrieved":
      logLine(`✓ Found ${event.comps.length} comparable listings:`);
      event.comps.forEach(c => {
        logLine(`   • $${c.price.toLocaleString()} — ${c.summary}...`);
      });
      break;
    case "agents_started":
      logLine("🤖 Pricing agents running...");
      break;
    case "specialist_done":
      logLine(`✓ Fine-tuned model estimate: $${event.price.toLocaleString()}`);
      break;
    case "frontier_done":
      logLine(`✓ RAG-augmented estimate: $${event.price.toLocaleString()}`);
      break;
    case "error":
      document.getElementById("error").textContent = event.message;
      break;
    case "final":
      renderResult(event);
      break;
  }
}

function renderResult(data) {
  const resultDiv = document.getElementById("result");
  const askingFmt = data.asking_price.toLocaleString(undefined, { style: "currency", currency: "USD" });
  const estimatedFmt = data.estimated_price.toLocaleString(undefined, { style: "currency", currency: "USD" });

  resultDiv.innerHTML = `
    <div class="row"><span>Asking:</span><span>${askingFmt}</span></div>
    <div class="row"><span>Estimated fair price:</span><span>${estimatedFmt}</span></div>
    <div class="verdict ${data.verdict}">${data.verdict} (${data.delta_pct > 0 ? "+" : ""}${data.delta_pct}%)</div>
  `;
}
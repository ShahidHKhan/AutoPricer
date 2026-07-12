const MODAL_ENDPOINT = "https://shahidhkhan--auto-pricer-service-pricingendpoint-web.modal.run/estimate";

document.getElementById("checkBtn").addEventListener("click", async () => {
  const resultDiv = document.getElementById("result");
  const errorDiv = document.getElementById("error");
  const btn = document.getElementById("checkBtn");

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

    if (!response.ok) {
      throw new Error(`Server error (${response.status})`);
    }

    const data = await response.json();
    renderResult(data);

  } catch (err) {
    errorDiv.textContent = err.message || "Something went wrong.";
  } finally {
    btn.disabled = false;
    btn.textContent = "Check This Listing";
  }
});

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
"use strict";

const form = document.getElementById("ask-form");
const questionEl = document.getElementById("question");
const askBtn = document.getElementById("ask");
const pdfBtn = document.getElementById("pdf");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const answerEl = document.getElementById("answer");
const limitationsEl = document.getElementById("limitations");
const evidenceEl = document.getElementById("evidence");
const relatedEl = document.getElementById("related");

let lastQuestion = "";

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function text(tag, cls, value) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  node.textContent = value == null ? "" : String(value);
  return node;
}

function metaLine(item) {
  const bits = [];
  if (item.label) bits.push(item.label);
  if (item.category) bits.push(item.category);
  if (item.source_doc) bits.push("doc: " + item.source_doc);
  if (item.section) bits.push("section: " + item.section);
  if (item.heading) bits.push(item.heading);
  if (item.court) bits.push(item.court);
  if (item.date) bits.push(item.date);
  if (typeof item.retrieval_score === "number") {
    bits.push("score: " + item.retrieval_score.toFixed(4));
  }
  return bits.join("  ·  ");
}

function card(item) {
  const wrapper = document.createElement("div");
  wrapper.className = "card";

  const meta = document.createElement("div");
  meta.className = "meta";
  if (item.label) {
    const label = text("span", "label", item.label);
    meta.appendChild(label);
  }
  meta.appendChild(document.createTextNode(metaLine({ ...item, label: null })));
  wrapper.appendChild(meta);

  wrapper.appendChild(text("div", "text", item.text || "(no text available)"));
  return wrapper;
}

function render(data) {
  answerEl.textContent = data.answer || "";

  limitationsEl.replaceChildren();
  (data.limitations || []).forEach((item) => limitationsEl.appendChild(text("li", null, item)));

  evidenceEl.replaceChildren();
  if (!data.evidence || data.evidence.length === 0) {
    evidenceEl.appendChild(text("p", "meta", "No cited passages were returned."));
  } else {
    data.evidence.forEach((item, i) => evidenceEl.appendChild(card({ ...item, label: "S" + (i + 1) })));
  }

  relatedEl.replaceChildren();
  if (!data.related_cases || data.related_cases.length === 0) {
    relatedEl.appendChild(text("p", "meta", "No related cases were found."));
  } else {
    data.related_cases.forEach((item) => relatedEl.appendChild(card(item)));
  }

  show(resultEl);
  hide(errorEl);
}

function showError(message) {
  errorEl.textContent = message;
  show(errorEl);
  hide(resultEl);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionEl.value.trim();
  if (!question) {
    showError("Please enter a legal question.");
    return;
  }

  askBtn.disabled = true;
  statusEl.textContent = "Retrieving evidence and generating an answer…";
  hide(errorEl);

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      let detail = "Request failed (" + response.status + ").";
      try {
        const body = await response.json();
        if (body && body.detail) detail = typeof body.detail === "string"
          ? body.detail
          : "The request could not be processed (validation error).";
      } catch (_) { /* keep default */ }
      throw new Error(detail);
    }

    render(await response.json());
    lastQuestion = question;
    pdfBtn.disabled = false;
    statusEl.textContent = "";
  } catch (err) {
    showError(err && err.message ? err.message : "Something went wrong.");
    statusEl.textContent = "";
  } finally {
    askBtn.disabled = false;
  }
});

pdfBtn.addEventListener("click", async () => {
  if (!lastQuestion) return;
  pdfBtn.disabled = true;
  statusEl.textContent = "Preparing PDF…";

  try {
    const response = await fetch("/chat/pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: lastQuestion }),
    });
    if (!response.ok) {
      throw new Error("PDF export failed (" + response.status + ").");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "legalrag-answer.pdf";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    statusEl.textContent = "";
  } catch (err) {
    showError(err && err.message ? err.message : "PDF export failed.");
    statusEl.textContent = "";
  } finally {
    pdfBtn.disabled = false;
  }
});

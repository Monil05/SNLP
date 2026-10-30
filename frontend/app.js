"use strict";

const EXAMPLE = {
  student_text: "B.Tech in Data Science. Skills: Python, SQL, pandas. Projects: Built reporting dashboards and cleaned business datasets. Academic record: NLP: 35%; Chemistry: 35%.",
  job_descriptions: [
    {
      job_title: "Business Intelligence Analyst",
      job_text: "Despite the title, the central technical domain of this role is Natural Language Processing (NLP). The work is built around RAG systems and NLP model evaluation. Deep expertise in NLP is required. Strong knowledge of RAG is mandatory. Candidates use Python, pandas, and SQL to process data. Knowledge of Chemistry is nice to have."
    },
    {
      job_title: "NLP Research Engineer",
      job_text: "This is a business intelligence role focused on Power BI dashboards, SQL reporting, and sales trends. NLP expertise is not required. Knowledge of Chemistry is not required but good to have."
    },
    {
      job_title: "Data Operations Analyst",
      job_text: "Clean, validate, transform, and maintain datasets using Python, pandas, and SQL. Data-quality checks and processing automation are required. Knowledge of Chemistry is not required."
    }
  ]
};

const $ = (id) => document.getElementById(id);
let manualSearchVersion = 0;
const MODEL_STATS = [
  { metric: "precision_at_5", baseline: 0.8625, core: 0.875, absolute_change: 0.0125, percentage_change: 1.4492753623 },
  { metric: "recall_at_5", baseline: 0.0924504352, core: 0.0938079263, absolute_change: 0.0013574911, percentage_change: 1.4683447749 },
  { metric: "mrr", baseline: 0.890625, core: 0.96875, absolute_change: 0.078125, percentage_change: 8.7719298246 }
];
const elements = {
  form: $("analysisForm"), studentText: $("studentText"), jobsList: $("jobsList"),
  studentUploadButton: $("studentUploadButton"), studentFileInput: $("studentFileInput"),
  studentUploadStatus: $("studentUploadStatus"),
  addJobButton: $("addJobButton"), loadExampleButton: $("loadExampleButton"),
  runButton: $("runButton"), runButtonLabel: $("runButtonLabel"),
  jobModeHint: $("jobModeHint"), jobCount: $("jobCount"), emptyJobsNote: $("emptyJobsNote"),
  formError: $("formError"), loadingState: $("loadingState"), loadingTitle: $("loadingTitle"),
  loadingDescription: $("loadingDescription"), resultsSection: $("resultsSection"),
  resultsTitle: $("resultsTitle"), resultsSummary: $("resultsSummary"), resultsList: $("resultsList"),
  clearResultsButton: $("clearResultsButton"), researcherSearchForm: $("researcherSearchForm"),
  researcherQuery: $("researcherQuery"), researcherSearchButton: $("researcherSearchButton"),
  clearResearcherSearchButton: $("clearResearcherSearchButton"),
  researcherSearchStatus: $("researcherSearchStatus"), researcherSearchResults: $("researcherSearchResults"),
  statsGrid: $("statsGrid")
};

function currentMode() {
  const count = elements.jobsList.querySelectorAll(".job-card").length;
  if (count === 0) return "recommend";
  if (count === 1) return "match";
  return "compare";
}

function modeText(mode) {
  if (mode === "match") {
    return {
      button: "Find match score",
      loadingTitle: "Calculating the match score…",
      loadingDescription: "The student profile is being compared with this job description.",
      resultsTitle: "Match result"
    };
  }
  if (mode === "compare") {
    return {
      button: "Compare jobs",
      loadingTitle: "Comparing job descriptions…",
      loadingDescription: "Each role is being scored against the student profile.",
      resultsTitle: "Job comparison"
    };
  }
  return {
    button: "Recommend jobs",
    loadingTitle: "Finding suitable roles…",
    loadingDescription: "The student profile is being compared with jobs in the dataset.",
    resultsTitle: "Recommended roles"
  };
}

function updateModeUI() {
  const cards = elements.jobsList.querySelectorAll(".job-card");
  const count = cards.length;
  const mode = currentMode();
  const copy = modeText(mode);
  elements.runButtonLabel.textContent = copy.button;
  elements.jobModeHint.textContent = count === 0
    ? "No jobs added. We'll recommend suitable roles from the job dataset."
    : count === 1
      ? "One job added. We'll calculate a match score for that role."
      : `${count} jobs added. We'll score and rank them against the student profile.`;
  elements.jobCount.textContent = count === 0
    ? "No job descriptions added"
    : `${count} job description${count === 1 ? "" : "s"} added`;
  elements.emptyJobsNote.hidden = count !== 0;
  elements.addJobButton.disabled = count >= 20;
  elements.addJobButton.title = count >= 20 ? "Maximum of 20 job descriptions" : "Add a job description";
  [...cards].forEach((card, index) => {
    const number = index + 1;
    card.querySelector(".job-number").textContent = String(number);
    card.querySelector(".job-heading-text").textContent = `Job ${number}`;
    card.querySelector(".job-id").textContent = `ID ${number}`;
    card.querySelector(".remove-job").title = `Remove Job ${number}`;
  });
}

async function extractUploadedFile(file, textarea, status) {
  if (!file) return;
  status.textContent = `Extracting text from ${file.name}…`;
  status.classList.remove("upload-error");
  try {
    const body = new FormData();
    body.append("file", file, file.name);
    let response = await fetch("/extract-document", { method: "POST", body });
    let usedApiFallback = false;
    if (response.status === 405 || response.status === 404) {
      const apiBody = new FormData();
      apiBody.append("file", file, file.name);
      usedApiFallback = true;
      response = await fetch("http://127.0.0.1:8000/v1/extract-document", { method: "POST", body: apiBody });
    }
    const data = await response.json().catch(() => ({}));
    if (usedApiFallback && response.status === 404) {
      throw new Error("The Career NLP backend needs to be restarted to enable document uploads.");
    }
    if (!response.ok && (response.status === 405 || response.status === 0)) {
      throw new Error("Could not reach the document extractor. Make sure the Career NLP backend is running on port 8000.");
    }
    if (!response.ok) throw new Error(data.detail || data.error?.message || `Could not extract this file (HTTP ${response.status}).`);
    const existing = textarea.value.trim();
    textarea.value = existing ? `${existing}\n\n${data.text}` : data.text;
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    status.textContent = `Added text from ${data.filename}. You can edit it below.`;
  } catch (error) {
    status.textContent = error instanceof TypeError
      ? "Could not connect to the document extractor. Check that the Career NLP backend is running on port 8000."
      : error.message || "Could not extract text from this file.";
    status.classList.add("upload-error");
  }
}

function addJob(job = {}) {
  if (elements.jobsList.querySelectorAll(".job-card").length >= 20) return;

  const card = document.createElement("article");
  card.className = "job-card";

  const header = document.createElement("div");
  header.className = "job-card-header";
  const titleLine = document.createElement("div");
  titleLine.className = "job-card-title";
  const number = document.createElement("span");
  number.className = "job-number";
  const heading = document.createElement("span");
  heading.className = "job-heading-text";
  const id = document.createElement("span");
  id.className = "job-id";
  titleLine.append(number, heading, id);

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "remove-job";
  remove.textContent = "Remove";
  remove.addEventListener("click", () => {
    card.remove();
    updateModeUI();
  });
  header.append(titleLine, remove);

  const body = document.createElement("div");
  body.className = "job-card-body";

  const titleLabel = document.createElement("label");
  titleLabel.className = "field";
  const titleText = document.createElement("span");
  titleText.textContent = "Job title (optional)";
  const titleInput = document.createElement("input");
  titleInput.type = "text";
  titleInput.className = "job-title-input";
  titleInput.maxLength = 200;
  titleInput.placeholder = "e.g. Data Operations Analyst";
  titleInput.value = job.job_title || "";
  titleInput.autocomplete = "off";
  titleLabel.append(titleText, titleInput);

  const descriptionLabel = document.createElement("label");
  descriptionLabel.className = "field";
  const descriptionText = document.createElement("span");
  descriptionText.textContent = "Job description";
  const description = document.createElement("textarea");
  description.className = "job-description-input";
  description.rows = 5;
  description.maxLength = 50000;
  description.placeholder = "Paste the job description, including required and preferred skills…";
  description.value = job.job_text || "";
  const uploadRow = document.createElement("div");
  uploadRow.className = "document-upload-row job-document-upload-row";
  const uploadButton = el("button", "button button-secondary button-small", "Upload job description");
  uploadButton.type = "button";
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = ".pdf,.doc,.docx,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document";
  fileInput.hidden = true;
  const uploadStatus = el("span", "document-upload-status", "PDF or Word, up to 5 MB");
  uploadStatus.setAttribute("role", "status");
  uploadStatus.setAttribute("aria-live", "polite");
  uploadButton.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    extractUploadedFile(fileInput.files[0], description, uploadStatus);
    fileInput.value = "";
  });
  uploadRow.append(uploadButton, fileInput, uploadStatus);
  descriptionLabel.append(descriptionText, description, uploadRow);
  body.append(titleLabel, descriptionLabel);
  card.append(header, body);
  elements.jobsList.append(card);
  updateModeUI();
}

function showError(message) {
  elements.formError.textContent = message;
  elements.formError.hidden = !message;
}

function collectSubmission() {
  const studentText = elements.studentText.value.trim();
  if (!studentText) throw new Error("Enter the student profile first.");

  const cards = [...elements.jobsList.querySelectorAll(".job-card")];
  if (cards.length > 20) throw new Error("You can compare up to 20 job descriptions at a time.");

  const jobs = cards.map((card, index) => {
    const jobText = card.querySelector(".job-description-input").value.trim();
    const jobTitle = card.querySelector(".job-title-input").value.trim();
    if (!jobText) {
      throw new Error(`Job ${index + 1} has no description. Fill it in or remove that job to continue.`);
    }
    const job = { job_id: String(index + 1), job_text: jobText };
    if (jobTitle) job.job_title = jobTitle;
    return { ...job, _displayTitle: jobTitle };
  });

  const mode = jobs.length === 0 ? "recommend" : jobs.length === 1 ? "match" : "compare";
  let endpoint;
  let request;

  if (mode === "recommend") {
    endpoint = "/api/v1/recommend";
    request = { student_text: studentText, top_k: 3, include_details: true };
  } else if (mode === "match") {
    endpoint = "/api/v1/match";
    request = { student_text: studentText, job_text: jobs[0].job_text, include_details: true };
  } else {
    endpoint = "/api/v1/compare-jobs";
    request = {
      student_text: studentText,
      job_descriptions: jobs.map(({ job_id, job_title, job_text }) => {
        const item = { job_id, job_text };
        if (job_title) item.job_title = job_title;
        return item;
      }),
      include_details: true
    };
  }

  return { mode, endpoint, request, jobs };
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function formatNumber(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function renderModelStats(metrics) {
  const labels = {
    precision_at_5: "Precision@5",
    recall_at_5: "Recall@5",
    mrr: "Mean Reciprocal Rank (MRR)"
  };
  elements.statsGrid.replaceChildren();
  metrics.forEach((metric) => {
    const card = el("article", "stats-card");
    card.append(el("h3", "", labels[metric.metric] || metric.metric));

    const comparison = el("div", "stats-comparison");
    const baseline = el("div", "stats-model baseline-model");
    baseline.append(el("span", "stats-model-label", "Baseline · TF-IDF"));
    baseline.append(el("strong", "stats-value", `${formatNumber(metric.baseline * 100)}%`));

    const core = el("div", "stats-model core-model");
    core.append(el("span", "stats-model-label", "Core pipeline · Sentence Transformer hybrid"));
    core.append(el("strong", "stats-value", `${formatNumber(metric.core * 100)}%`));
    comparison.append(baseline, core);
    card.append(comparison);

    const change = Number(metric.absolute_change) * 100;
    const sign = change >= 0 ? "+" : "";
    card.append(el("p", "stats-change", `${sign}${formatNumber(change)} percentage points (${sign}${formatNumber(metric.percentage_change)}% relative)`));
    elements.statsGrid.append(card);
  });
  elements.statsGrid.hidden = false;
}

function showView(viewId) {
  document.querySelectorAll(".app-view").forEach((view) => {
    view.hidden = view.id !== viewId;
  });
  document.querySelectorAll(".nav-item").forEach((button) => {
    const selected = button.dataset.view === viewId;
    button.classList.toggle("active", selected);
    if (selected) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  if (viewId === "statsView") renderModelStats(MODEL_STATS);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderChips(target, items, className = "neutral", emptyText = "None detected") {
  if (!Array.isArray(items) || items.length === 0) {
    target.append(el("span", "empty-note", emptyText));
    return;
  }
  const list = el("div", "pill-list");
  items.forEach((item) => list.append(el("span", `pill ${className}`, item)));
  target.append(list);
}

function addResultGroup(parent, title, items, className, emptyText) {
  const group = el("section", "result-group");
  group.append(el("h4", "", title));
  renderChips(group, items, className, emptyText);
  parent.append(group);
}

function safeExternalUrl(value) {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

function renderResearchers(target, researchers) {
  target.replaceChildren();
  if (!Array.isArray(researchers) || researchers.length === 0) {
    target.append(el("p", "empty-note", "No researchers were found for this topic."));
    return;
  }

  researchers.forEach((researcher) => {
    const card = el("article", "researcher-card");
    const heading = el("div", "researcher-card-heading");
    const name = el("h5", "", researcher.name || "Researcher");
    const authorUrl = safeExternalUrl(researcher.author_id || "");
    if (authorUrl) {
      const link = el("a", "researcher-profile-link", "OpenAlex profile");
      link.href = authorUrl;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      name.append(" ", link);
    }
    const score = Number(researcher.score);
    heading.append(name);
    if (Number.isFinite(score)) {
      heading.append(el("span", "researcher-score", `Publication match ${formatNumber(score * 100, 1)}%`));
    }
    card.append(heading);

    if (researcher.confidence?.level) {
      card.append(el("p", "researcher-confidence", `Ranking confidence: ${researcher.confidence.level}`));
    }
    const papers = Array.isArray(researcher.evidence) ? researcher.evidence : [];
    if (papers.length) {
      const paperList = el("div", "researcher-publications");
      papers.slice(0, 2).forEach((paper, index) => {
        const details = [paper.year, paper.similarity !== undefined ? `match ${formatNumber(Number(paper.similarity) * 100, 1)}%` : ""]
          .filter(Boolean).join(" · ");
        const item = el("div", "researcher-publication");
        item.append(el("span", "researcher-publication-label", index === 0 ? "Most relevant paper" : "Related paper"));
        item.append(el("span", "researcher-paper-title", paper.title || "Supporting publication"));
        if (details) item.append(el("small", "", details));
        paperList.append(item);
      });
      card.append(paperList);
    }
    target.append(card);
  });
}

async function requestResearchers(query, topK = 3) {
  const response = await fetch("/api/group7/v1/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Accept": "application/json" },
    body: JSON.stringify({ query, top_k: topK, include_evidence: true, paper_top_k: 100 })
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(apiErrorMessage(data, response.status));
  return Array.isArray(data.results) ? data.results : [];
}

function addLearningResearch(card, topics) {
  const actions = el("div", "learning-research-actions");
  const button = el("button", "button button-quiet button-small learning-research-button", "Find teachers for recommended learning");
  button.type = "button";
  const clearButton = el("button", "button button-quiet button-small learning-research-clear", "Clear");
  clearButton.type = "button";
  actions.append(button, clearButton);
  const results = el("div", "learning-research-results");
  card.append(actions, results);
  let requestNumber = 0;

  clearButton.addEventListener("click", () => {
    requestNumber += 1;
    results.replaceChildren();
    button.disabled = false;
    button.textContent = "Find teachers for recommended learning";
  });

  button.addEventListener("click", async () => {
    const currentRequest = ++requestNumber;
    button.disabled = true;
    button.textContent = "Searching Group 7…";
    results.replaceChildren(el("p", "empty-note", "Searching publications for each learning topic…"));
    try {
      const matches = await Promise.all(topics.map(async (topic) => ({ topic, researchers: await requestResearchers(topic, 3) })));
      if (currentRequest !== requestNumber) return;
      results.replaceChildren();
      matches.forEach(({ topic, researchers }) => {
        const topicSection = el("section", "learning-topic-results");
        topicSection.append(el("h4", "", topic));
        const researcherList = el("div", "researcher-results");
        renderResearchers(researcherList, researchers);
        topicSection.append(researcherList);
        results.append(topicSection);
      });
      button.textContent = "Refresh teacher matches";
    } catch (error) {
      if (currentRequest === requestNumber) {
        results.replaceChildren(el("p", "form-error", error.message || "Group 7 could not be reached."));
        button.textContent = "Retry researcher search";
      }
    } finally {
      if (currentRequest === requestNumber) button.disabled = false;
    }
  });
}

function renderJobResult(job, isTopResult, mode) {
  const match = job.match || job;
  const score = Math.max(0, Math.min(100, Number(match.match_score) || 0));
  const card = el("article", `result-card${isTopResult ? " recommended" : ""}`);
  const top = el("div", "result-top");
  top.append(el("div", "rank-badge", `#${job.rank ?? 1}`));

  const titleArea = el("div", "result-title-area");
  const titleLine = el("h3", "", job.job_title || `Job ${job.job_id || ""}`);
  if (isTopResult && mode !== "match") {
    titleLine.append(el("span", "recommended-label", "BEST MATCH"));
  }
  titleArea.append(titleLine, el("p", "", `Job ID ${job.job_id || "1"}${job.category ? ` · ${job.category}` : ""}`));

  const scoreBlock = el("div", "score-block");
  scoreBlock.append(
    el("div", "score-value", `${formatNumber(match.match_score)}%`),
    el("div", "score-label", "Job match")
  );
  top.append(titleArea, scoreBlock);

  const bar = el("div", "score-bar");
  const fill = el("div", "score-bar-fill");
  fill.style.width = `${score}%`;
  bar.append(fill);
  card.append(top, bar);

  const content = el("div", "result-content");
  const groups = el("div", "result-groups");
  addResultGroup(groups, "Matched skills", match.matched_skills || [], "", "No matched skills listed");
  addResultGroup(groups, "Required gaps", match.missing_weak_skills || [], "missing", "No required gaps detected");
  addResultGroup(groups, "Recommended learning", match.recommended_learning || [], "learning", "No learning suggestions");
  content.append(groups);
  card.append(content);
  const learning = [...new Set((match.recommended_learning || []).map((item) => String(item).trim()).filter(Boolean))];
  if (learning.length) addLearningResearch(card, learning);
  return card;
}

function renderResults(data, submission) {
  const { mode, jobs } = submission;
  let ranked;
  if (mode === "match") {
    ranked = [{ rank: 1, job_id: "1", job_title: jobs[0]._displayTitle || "Provided job description", match: data }];
  } else {
    const recommended = data.recommended_job;
    const alternatives = Array.isArray(data.alternatives) ? data.alternatives : [];
    ranked = recommended ? [recommended, ...alternatives] : alternatives;
  }

  elements.resultsList.replaceChildren();
  const copy = modeText(mode);
  elements.resultsTitle.textContent = copy.resultsTitle;
  if (mode === "recommend") {
    elements.resultsSummary.textContent = `${ranked.length} roles recommended from the job dataset.`;
  } else if (mode === "match") {
    elements.resultsSummary.textContent = `Match score for ${jobs[0]._displayTitle || "the provided job description"}.`;
  } else {
    elements.resultsSummary.textContent = `${ranked.length} job descriptions ranked by match score.`;
  }
  ranked.forEach((job, index) => {
    elements.resultsList.append(renderJobResult(job, index === 0 && mode !== "match", mode));
  });
  elements.resultsSection.hidden = false;
}

function apiErrorMessage(data, status) {
  if (typeof data?.detail === "string") return data.detail;
  if (typeof data?.message === "string") return data.message;
  if (Array.isArray(data?.detail)) {
    return data.detail.map((item) => `${Array.isArray(item.loc) ? item.loc.join(" → ") : "Request"}: ${item.msg || "Invalid value"}`).join("; ");
  }
  if (typeof data?.error?.message === "string") return data.error.message;
  return `The request failed (HTTP ${status}). Check the input and try again.`;
}

async function runAnalysis(event) {
  event.preventDefault();
  manualSearchVersion += 1;
  showError("");

  let submission;
  try {
    submission = collectSubmission();
  } catch (error) {
    showError(error.message || "Check the form and try again.");
    return;
  }

  const copy = modeText(submission.mode);
  elements.runButton.disabled = true;
  elements.runButtonLabel.textContent = "Working…";
  elements.loadingTitle.textContent = copy.loadingTitle;
  elements.loadingDescription.textContent = copy.loadingDescription;
  elements.loadingState.hidden = false;
  elements.resultsSection.hidden = true;

  try {
    const response = await fetch(submission.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(submission.request)
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(apiErrorMessage(data, response.status));
    renderResults(data, submission);
    elements.resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showError(error.message || "The request could not be completed.");
  } finally {
    elements.runButton.disabled = false;
    elements.runButtonLabel.textContent = modeText(currentMode()).button;
    elements.loadingState.hidden = true;
  }
}

async function searchResearchers(event) {
  event.preventDefault();
  manualSearchVersion += 1;
  const searchVersion = manualSearchVersion;
  const query = elements.researcherQuery.value.trim();
  elements.researcherSearchResults.replaceChildren();
  elements.researcherSearchStatus.textContent = "";
  if (!query) {
    elements.researcherSearchStatus.textContent = "Enter a topic to search.";
    return;
  }

  elements.researcherSearchButton.disabled = true;
  elements.researcherSearchStatus.textContent = `Searching Group 7 publications for “${query}”…`;
  try {
    const researchers = await requestResearchers(query, 50);
    if (searchVersion !== manualSearchVersion) return;
    renderResearchers(elements.researcherSearchResults, researchers);
    elements.researcherSearchStatus.textContent = `Showing ${researchers.length} researcher matches for “${query}” (Group 7 returns up to 50).`;
  } catch (error) {
    if (searchVersion === manualSearchVersion) {
      elements.researcherSearchStatus.textContent = error.message || "Group 7 could not be reached.";
    }
  } finally {
    if (searchVersion === manualSearchVersion) elements.researcherSearchButton.disabled = false;
  }
}

function clearResearcherSearch() {
  manualSearchVersion += 1;
  elements.researcherQuery.value = "";
  elements.researcherSearchResults.replaceChildren();
  elements.researcherSearchStatus.textContent = "";
  elements.researcherSearchButton.disabled = false;
}

function loadExample() {
  elements.studentText.value = EXAMPLE.student_text;
  elements.jobsList.replaceChildren();
  EXAMPLE.job_descriptions.forEach((job) => addJob(job));
  showError("");
  clearResults();
  updateModeUI();
}

function clearResults() {
  elements.resultsSection.hidden = true;
  elements.resultsList.replaceChildren();
}

elements.addJobButton.addEventListener("click", () => {
  addJob();
  showError("");
});
elements.studentUploadButton.addEventListener("click", () => elements.studentFileInput.click());
elements.studentFileInput.addEventListener("change", () => {
  extractUploadedFile(elements.studentFileInput.files[0], elements.studentText, elements.studentUploadStatus);
  elements.studentFileInput.value = "";
});
elements.loadExampleButton.addEventListener("click", loadExample);
elements.form.addEventListener("submit", runAnalysis);
elements.clearResultsButton.addEventListener("click", clearResults);
elements.researcherSearchForm.addEventListener("submit", searchResearchers);
elements.clearResearcherSearchButton.addEventListener("click", clearResearcherSearch);
elements.studentText.addEventListener("input", () => showError(""));
document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => showView(button.dataset.view));
});

elements.studentText.value = "";
updateModeUI();

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
const elements = {
  form: $("analysisForm"), studentText: $("studentText"), jobsList: $("jobsList"),
  addJobButton: $("addJobButton"), loadExampleButton: $("loadExampleButton"),
  runButton: $("runButton"), runButtonLabel: $("runButtonLabel"),
  jobModeHint: $("jobModeHint"), jobCount: $("jobCount"), emptyJobsNote: $("emptyJobsNote"),
  formError: $("formError"), loadingState: $("loadingState"), loadingTitle: $("loadingTitle"),
  loadingDescription: $("loadingDescription"), resultsSection: $("resultsSection"),
  resultsTitle: $("resultsTitle"), resultsSummary: $("resultsSummary"), resultsList: $("resultsList"),
  clearResultsButton: $("clearResultsButton")
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
  descriptionLabel.append(descriptionText, description);
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
  if (Array.isArray(data?.detail)) {
    return data.detail.map((item) => `${Array.isArray(item.loc) ? item.loc.join(" → ") : "Request"}: ${item.msg || "Invalid value"}`).join("; ");
  }
  if (typeof data?.error?.message === "string") return data.error.message;
  return `The request failed (HTTP ${status}). Check the input and try again.`;
}

async function runAnalysis(event) {
  event.preventDefault();
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
elements.loadExampleButton.addEventListener("click", loadExample);
elements.form.addEventListener("submit", runAnalysis);
elements.clearResultsButton.addEventListener("click", clearResults);
elements.studentText.addEventListener("input", () => showError(""));

elements.studentText.value = "";
updateModeUI();

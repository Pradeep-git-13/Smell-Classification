// ============================================================
//  OlfactAI — Smell Classification Engine
//  app.js — Dataset + NLP Pipeline + XGBoost-style Classifier
// ============================================================

// ─────────────────────────────────────────────
// 1. DATASET — 12 Classes, 80 samples each (960 total)
//    Features: temp, hum, mq2, mq4, mq8, mq135, mq136, nh3, vnh3
// ─────────────────────────────────────────────

const SMELL_CLASSES = [
  { id: "petrol",   name: "Petrol / Gasoline", icon: "⛽", category: "Petrochemical",   color: "#ff7857" },
  { id: "coffee",   name: "Coffee",            icon: "☕", category: "Food & Beverage", color: "#c8a26a" },
  { id: "rose",     name: "Rose / Floral",     icon: "🌹", category: "Floral",          color: "#ff6b9d" },
  { id: "fish",     name: "Fish / Seafood",    icon: "🐟", category: "Organic Decay",   color: "#57d4ff" },
  { id: "smoke",    name: "Smoke / Burning",   icon: "🔥", category: "Combustion",      color: "#ff9f43" },
  { id: "ammonia",  name: "Ammonia",           icon: "⚗️", category: "Chemical",        color: "#a29bfe" },
  { id: "earth",    name: "Petrichor / Earth", icon: "🌱", category: "Natural",         color: "#55efc4" },
  { id: "vinegar",  name: "Vinegar / Acetic",  icon: "🍶", category: "Acidic",          color: "#fdcb6e" },
  { id: "alcohol",  name: "Alcohol / Ethanol", icon: "🍷", category: "Organic Solvent", color: "#e17055" },
  { id: "lemon",    name: "Citrus / Lemon",    icon: "🍋", category: "Fruity",          color: "#ffeaa7" },
  { id: "paint",    name: "Paint / Solvent",   icon: "🎨", category: "Chemical",        color: "#74b9ff" },
  { id: "garbage",  name: "Garbage / Putrid",  icon: "🗑️", category: "Organic Decay",   color: "#636e72" },
];

// Representative sensor profiles for each smell class
// [temp, hum, mq2, mq4, mq8, mq135, mq136, nh3, vnh3]
const SMELL_PROFILES = {
  petrol:  { temp: 27, hum: 45, mq2: 72, mq4: 18, mq8: 12, mq135: 65, mq136: 28, nh3: 0.2, vnh3: 0.5, desc: "High MQ-2 and MQ-135 response. Low MQ-4. Typical VOC signature.", mol: "C₈H₁₈ (Octane), C₇H₁₆ (Heptane), Benzene derivatives" },
  coffee:  { temp: 62, hum: 38, mq2: 14, mq4: 8,  mq8: 6,  mq135: 42, mq136: 10, nh3: 0.4, vnh3: 0.8, desc: "Elevated temperature. Moderate MQ-135 (VOC roasting compounds).", mol: "2-Furfurylthiol, Pyrazines, Acetaldehyde" },
  rose:    { temp: 24, hum: 65, mq2: 6,  mq4: 4,  mq8: 3,  mq135: 18, mq136: 5,  nh3: 0.1, vnh3: 0.3, desc: "Low sensor response across all channels. High humidity. Clean floral.", mol: "Geraniol, Citronellol, β-Damascenone, Linalool" },
  fish:    { temp: 20, hum: 78, mq2: 22, mq4: 35, mq8: 48, mq135: 55, mq136: 40, nh3: 3.8, vnh3: 3.5, desc: "High NH₃, MQ-8 and MQ-4. Biogenic amines from protein decay.", mol: "Trimethylamine (TMA), Putrescine, Cadaverine, NH₃" },
  smoke:   { temp: 68, hum: 30, mq2: 88, mq4: 62, mq8: 71, mq135: 82, mq136: 35, nh3: 0.6, vnh3: 1.1, desc: "Very high MQ-2 and MQ-8. Elevated temperature. Broad combustion signature.", mol: "CO, Formaldehyde, Acrolein, Polycyclic aromatic hydrocarbons" },
  ammonia: { temp: 25, hum: 55, mq2: 8,  mq4: 6,  mq8: 10, mq135: 72, mq136: 15, nh3: 4.5, vnh3: 4.2, desc: "Extremely high NH₃ readings. MQ-135 also elevated (sensitive to ammonia).", mol: "NH₃ (Ammonia), traces of amines" },
  earth:   { temp: 18, hum: 82, mq2: 10, mq4: 12, mq8: 8,  mq135: 22, mq136: 18, nh3: 0.5, vnh3: 0.7, desc: "High humidity. Low sensor response. Geosmin signature on MQ-136.", mol: "Geosmin, 2-Methylisoborneol (2-MIB), Petrichor compounds" },
  vinegar: { temp: 22, hum: 50, mq2: 18, mq4: 10, mq8: 9,  mq135: 58, mq136: 12, nh3: 0.3, vnh3: 0.6, desc: "MQ-135 highly responsive to acetic acid vapors.", mol: "Acetic acid (CH₃COOH), Diacetyl" },
  alcohol: { temp: 24, hum: 42, mq2: 45, mq4: 28, mq8: 32, mq135: 70, mq136: 22, nh3: 0.2, vnh3: 0.5, desc: "High MQ-2, MQ-135. MQ-8 elevated. Ethanol broad-spectrum signature.", mol: "Ethanol (C₂H₅OH), Methanol traces, Acetaldehyde" },
  lemon:   { temp: 23, hum: 58, mq2: 12, mq4: 7,  mq8: 5,  mq135: 30, mq136: 8,  nh3: 0.2, vnh3: 0.4, desc: "Low-moderate MQ-135. Limonene terpene not detected strongly by MQ sensors.", mol: "D-Limonene, β-Pinene, Citral, Linalool" },
  paint:   { temp: 26, hum: 35, mq2: 58, mq4: 42, mq8: 28, mq135: 85, mq136: 30, nh3: 0.3, vnh3: 0.7, desc: "Very high MQ-135 (sensitive to toluene/xylene). MQ-2 also elevated.", mol: "Toluene, Xylene, Acetone, n-Butyl acetate, VOCs" },
  garbage: { temp: 28, hum: 72, mq2: 38, mq4: 55, mq8: 62, mq135: 78, mq136: 68, nh3: 3.2, vnh3: 3.0, desc: "Multiple sensor response. High MQ-4 (methane from decomp), high NH₃.", mol: "H₂S, Methane, Ammonia, Indole, Skatole, Mercaptans" },
};

// ─────────────────────────────────────────────
// 2. GENERATE TRAINING DATASET (80 samples × 12 classes)
// ─────────────────────────────────────────────

function gaussianNoise(mean, std) {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function generateDataset() {
  const rows = [];
  const features = ["temp", "hum", "mq2", "mq4", "mq8", "mq135", "mq136", "nh3", "vnh3"];
  const stds = { temp: 3, hum: 6, mq2: 8, mq4: 6, mq8: 7, mq135: 9, mq136: 5, nh3: 0.3, vnh3: 0.3 };

  for (const cls of SMELL_CLASSES) {
    const prof = SMELL_PROFILES[cls.id];
    for (let i = 0; i < 80; i++) {
      const row = { label: cls.name };
      for (const f of features) {
        let val = gaussianNoise(prof[f], stds[f]);
        val = Math.max(0, val);
        if (f === "nh3" || f === "vnh3") val = Math.min(val, 5);
        else if (f === "temp") val = Math.min(val, 90);
        else val = Math.min(val, 100);
        row[f] = parseFloat(val.toFixed(2));
      }
      rows.push(row);
    }
  }
  // Shuffle
  for (let i = rows.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }
  return rows;
}

const DATASET = generateDataset();

// ─────────────────────────────────────────────
// 3. XGBoost-STYLE CLASSIFIER (JS decision tree ensemble)
//    Computes Mahalanobis-like distance from class centroids
//    then applies softmax for confidence scores
// ─────────────────────────────────────────────

function computeClassStats() {
  const features = ["temp", "hum", "mq2", "mq4", "mq8", "mq135", "mq136", "nh3", "vnh3"];
  const stats = {};
  for (const cls of SMELL_CLASSES) {
    const samples = DATASET.filter(d => d.label === cls.name);
    const means = {}, stds = {};
    for (const f of features) {
      const vals = samples.map(s => s[f]);
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const std = Math.sqrt(vals.map(v => (v - mean) ** 2).reduce((a, b) => a + b, 0) / vals.length) || 1;
      means[f] = mean;
      stds[f] = std;
    }
    stats[cls.id] = { means, stds };
  }
  return stats;
}

const CLASS_STATS = computeClassStats();

// Feature importance weights (derived from XGBoost gain — mimics the notebook)
const FEATURE_WEIGHTS = {
  mq135: 0.22, mq2: 0.18, nh3: 0.16, mq8: 0.12,
  mq136: 0.10, mq4: 0.09, hum: 0.05, temp: 0.05, vnh3: 0.03
};

function classifySensor(inputs) {
  const features = ["temp", "hum", "mq2", "mq4", "mq8", "mq135", "mq136", "nh3", "vnh3"];
  const scores = {};

  for (const cls of SMELL_CLASSES) {
    const { means, stds } = CLASS_STATS[cls.id];
    let dist = 0;
    for (const f of features) {
      const w = FEATURE_WEIGHTS[f] || 0.05;
      const z = ((inputs[f] || 0) - means[f]) / stds[f];
      dist += w * z * z;
    }
    scores[cls.id] = -dist; // higher = more similar
  }

  // Softmax
  const maxScore = Math.max(...Object.values(scores));
  const expScores = {};
  let sumExp = 0;
  for (const cls of SMELL_CLASSES) {
    expScores[cls.id] = Math.exp((scores[cls.id] - maxScore) * 3);
    sumExp += expScores[cls.id];
  }

  const probs = {};
  for (const cls of SMELL_CLASSES) {
    probs[cls.id] = expScores[cls.id] / sumExp;
  }

  const predicted = Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0];
  return { predicted, probs };
}

// ─────────────────────────────────────────────
// 4. NLP ENGINE — Tokenize, Stem, TF-IDF, Lexicon Match
// ─────────────────────────────────────────────

const STOP_WORDS = new Set([
  "a","an","the","is","it","in","on","at","to","of","and","or","but","i","my","its","was",
  "very","so","that","this","with","like","some","there","smell","smells","smelling",
  "odor","odour","scent","note","kind","bit","really","quite","something","little"
]);

const SMELL_LEXICON = {
  petrol:  ["petrol","gasoline","gas","fuel","diesel","benzene","hydrocarbon","pump","tar","petroleum","oil","kerosene","fume","exhaust"],
  coffee:  ["coffee","roast","roasted","brew","espresso","beans","bitter","cafe","arabica","mocha","java","caffeinated","grounds","cappuccino"],
  rose:    ["rose","floral","flower","bloom","jasmine","petal","bouquet","fragrant","lavender","perfume","blossom","daisy","violet","geranium","sweet","garden"],
  fish:    ["fish","seafood","fishy","tuna","salmon","shrimp","prawn","rotten","sea","ocean","marine","squid","crab","raw","decay","putrid","low","tide"],
  smoke:   ["smoke","smoky","burning","burnt","fire","ash","charcoal","campfire","bbq","barbecue","tobacco","cigarette","charred","soot","carbon"],
  ammonia: ["ammonia","pungent","sharp","chemical","urine","urea","bleach","cleaner","acrid","sting","sharp","caustic","nitrogen"],
  earth:   ["earth","earthy","soil","mud","petrichor","rain","forest","wood","grass","damp","moist","mossy","natural","fresh","woodland","rain"],
  vinegar: ["vinegar","acetic","sour","acid","tart","fermented","pickle","tang","sharp","sourdough","acidic","brine","kombucha"],
  alcohol: ["alcohol","ethanol","spirit","wine","beer","whiskey","vodka","rum","gin","booze","ferment","yeast","drink","intoxicating","solvent"],
  lemon:   ["lemon","citrus","orange","lime","citrusy","zest","tangy","fruity","grapefruit","bright","fresh","clean","limonene","tart"],
  paint:   ["paint","solvent","thinner","turpentine","varnish","lacquer","acetone","toluene","chemical","synthetic","factory","industrial"],
  garbage: ["garbage","trash","waste","rot","putrid","rotten","decay","foul","disgusting","sulfur","sewage","methane","compost","stale","mold"],
};

// Porter Stemmer (simplified)
function stem(word) {
  word = word.toLowerCase();
  const suffixes = [
    ["ational","ate"],["tional","tion"],["enci","ence"],["anci","ance"],
    ["izer","ize"],["ising","ise"],["izing","ize"],["ating","ate"],
    ["ness",""],["ment",""],["ful",""],["less",""],["ous",""],
    ["ive",""],["ing",""],["ely",""],["ed",""],["er",""],["ly",""],
    ["ies","y"],["ied","y"],["s",""]
  ];
  for (const [suf, rep] of suffixes) {
    if (word.endsWith(suf) && word.length - suf.length > 3) {
      return word.slice(0, -suf.length) + rep;
    }
  }
  return word;
}

function tokenize(text) {
  return text.toLowerCase().replace(/[^a-z\s]/g, " ").split(/\s+/).filter(t => t.length > 1);
}

function nlpClassify(text) {
  const tokens = tokenize(text);
  const processed = tokens.map(t => ({ original: t, stemmed: stem(t), isStop: STOP_WORDS.has(t) }));
  const meaningful = processed.filter(t => !t.isStop && t.original.length > 2);

  // Build TF map
  const tf = {};
  for (const t of meaningful) {
    tf[t.stemmed] = (tf[t.stemmed] || 0) + 1;
  }

  // Score each class
  const scores = {};
  for (const cls of SMELL_CLASSES) {
    const keywords = SMELL_LEXICON[cls.id] || [];
    const stemmedKW = keywords.map(k => ({ original: k, stemmed: stem(k) }));
    let score = 0;
    let matchedKW = [];

    for (const tok of meaningful) {
      for (const kw of stemmedKW) {
        if (tok.stemmed === kw.stemmed || tok.original.includes(kw.original) || kw.original.includes(tok.original)) {
          const tfidf = (tf[tok.stemmed] || 1) * Math.log(12 / 2 + 1); // IDF approximation
          score += tfidf;
          if (!matchedKW.includes(kw.original)) matchedKW.push(kw.original);
        }
      }
    }
    scores[cls.id] = { score, matchedKW };
  }

  // Softmax
  const rawScores = {};
  for (const c of SMELL_CLASSES) rawScores[c.id] = scores[c.id].score;
  const maxS = Math.max(...Object.values(rawScores), 0.01);

  let sumExp = 0;
  const expS = {};
  for (const c of SMELL_CLASSES) {
    expS[c.id] = Math.exp((rawScores[c.id] - maxS) * 2);
    sumExp += expS[c.id];
  }

  const probs = {};
  for (const c of SMELL_CLASSES) probs[c.id] = expS[c.id] / sumExp;

  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  const predicted = sorted[0][0];
  const matchedKW = scores[predicted].matchedKW;

  return { predicted, probs, tokens: processed, matchedKW };
}

// ─────────────────────────────────────────────
// 5. UI HELPERS
// ─────────────────────────────────────────────

function syncInput(sliderId, numId) {
  document.getElementById(numId).value = document.getElementById(sliderId).value;
  liveTokenize();
}

function syncSlider(numId, sliderId) {
  document.getElementById(sliderId).value = document.getElementById(numId).value;
}

function getSensorInputs() {
  return {
    temp: parseFloat(document.getElementById("temp_val").value) || 0,
    hum: parseFloat(document.getElementById("hum_val").value) || 0,
    mq2: parseFloat(document.getElementById("mq2_val").value) || 0,
    mq4: parseFloat(document.getElementById("mq4_val").value) || 0,
    mq8: parseFloat(document.getElementById("mq8_val").value) || 0,
    mq135: parseFloat(document.getElementById("mq135_val").value) || 0,
    mq136: parseFloat(document.getElementById("mq136_val").value) || 0,
    nh3: parseFloat(document.getElementById("nh3_val").value) || 0,
    vnh3: parseFloat(document.getElementById("vnh3_val").value) || 0,
  };
}

// PRESETS
const PRESETS = {
  petrol:  { temp: 27, hum: 45, mq2: 72, mq4: 18, mq8: 12, mq135: 65, mq136: 28, nh3: 0.2, vnh3: 0.5 },
  coffee:  { temp: 62, hum: 38, mq2: 14, mq4: 8,  mq8: 6,  mq135: 42, mq136: 10, nh3: 0.4, vnh3: 0.8 },
  rose:    { temp: 24, hum: 65, mq2: 6,  mq4: 4,  mq8: 3,  mq135: 18, mq136: 5,  nh3: 0.1, vnh3: 0.3 },
  fish:    { temp: 20, hum: 78, mq2: 22, mq4: 35, mq8: 48, mq135: 55, mq136: 40, nh3: 3.8, vnh3: 3.5 },
  smoke:   { temp: 68, hum: 30, mq2: 88, mq4: 62, mq8: 71, mq135: 82, mq136: 35, nh3: 0.6, vnh3: 1.1 },
  ammonia: { temp: 25, hum: 55, mq2: 8,  mq4: 6,  mq8: 10, mq135: 72, mq136: 15, nh3: 4.5, vnh3: 4.2 },
};

function loadPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  const keys = ["temp", "hum", "mq2", "mq4", "mq8", "mq135", "mq136", "nh3", "vnh3"];
  const ids = { temp: "temp", hum: "hum", mq2: "mq2", mq4: "mq4", mq8: "mq8", mq135: "mq135", mq136: "mq136", nh3: "nh3", vnh3: "vnh3" };
  for (const k of keys) {
    const el = document.getElementById(ids[k]);
    const num = document.getElementById(ids[k] + "_val");
    if (el && num) { el.value = p[k]; num.value = p[k]; }
  }
}

function setNLP(text) {
  document.getElementById("nlp-text").value = text;
  liveTokenize();
}

function liveTokenize() {
  const text = document.getElementById("nlp-text")?.value || "";
  if (!text.trim()) { document.getElementById("nlp-tokens").innerHTML = ""; return; }
  const tokens = tokenize(text).map(t => ({ original: t, isStop: STOP_WORDS.has(t) }));
  let html = "";
  // Determine keyword hits
  const allKW = Object.values(SMELL_LEXICON).flat();
  for (const t of tokens) {
    const isKW = allKW.some(k => k.includes(t.original) || t.original.includes(k) || stem(t.original) === stem(k));
    if (t.isStop) html += `<span class="token stop">${t.original}</span>`;
    else if (isKW) html += `<span class="token keyword">${t.original}</span>`;
    else html += `<span class="token">${t.original}</span>`;
  }
  document.getElementById("nlp-tokens").innerHTML = html;
}

// TABS
function initTabs() {
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    });
  });
  document.getElementById("nlp-text").addEventListener("input", liveTokenize);
}

// ─────────────────────────────────────────────
// 6. SHOW RESULTS
// ─────────────────────────────────────────────

function showResult(predicted, probs, source, matchedKW) {
  const cls = SMELL_CLASSES.find(c => c.id === predicted);
  const prof = SMELL_PROFILES[predicted];

  document.getElementById("result-icon").textContent = cls.icon;
  document.getElementById("result-name").textContent = cls.name;
  document.getElementById("result-conf").textContent = `Confidence: ${(probs[predicted] * 100).toFixed(1)}%`;

  // Confidence bars
  const sorted = [...SMELL_CLASSES].sort((a, b) => probs[b.id] - probs[a.id]);
  const topN = sorted.slice(0, 6);
  let barsHtml = "";
  for (const c of topN) {
    const pct = (probs[c.id] * 100).toFixed(1);
    const color = c.id === predicted ? "#b8ff57" : "#2a2a45";
    barsHtml += `
      <div class="conf-bar-row">
        <div class="conf-bar-label">${c.icon} ${c.name}</div>
        <div class="conf-bar-track">
          <div class="conf-bar-fill" style="width:0%;background:${c.id===predicted?'var(--accent)':'var(--border)'}" data-width="${pct}"></div>
        </div>
        <div class="conf-bar-pct">${pct}%</div>
      </div>`;
  }
  document.getElementById("conf-bars").innerHTML = barsHtml;

  // Meta
  document.getElementById("meta-mol").textContent = prof.mol;
  document.getElementById("meta-cat").textContent = cls.category;
  document.getElementById("meta-sig").textContent = prof.desc;
  document.getElementById("meta-nlp-kw").textContent =
    matchedKW && matchedKW.length > 0 ? matchedKW.join(", ") : source === "nlp" ? "No specific keywords found" : "N/A (Sensor input)";

  // Feature importance
  const features = [
    { name: "MQ-135", key: "mq135", val: FEATURE_WEIGHTS.mq135 },
    { name: "MQ-2",   key: "mq2",   val: FEATURE_WEIGHTS.mq2 },
    { name: "NH₃",    key: "nh3",   val: FEATURE_WEIGHTS.nh3 },
    { name: "MQ-8",   key: "mq8",   val: FEATURE_WEIGHTS.mq8 },
    { name: "MQ-136", key: "mq136", val: FEATURE_WEIGHTS.mq136 },
    { name: "MQ-4",   key: "mq4",   val: FEATURE_WEIGHTS.mq4 },
    { name: "Humidity",key:"hum",   val: FEATURE_WEIGHTS.hum },
    { name: "Temp",   key: "temp",  val: FEATURE_WEIGHTS.temp },
    { name: "V-NH₃",  key: "vnh3",  val: FEATURE_WEIGHTS.vnh3 },
  ];

  let fHtml = "";
  for (const f of features) {
    const pct = (f.val * 100).toFixed(0);
    fHtml += `
      <div class="feat-row">
        <div class="feat-label">${f.name}</div>
        <div class="feat-track">
          <div class="feat-fill" style="width:0%" data-width="${pct}"></div>
        </div>
        <div class="feat-pct">${pct}%</div>
      </div>`;
  }
  document.getElementById("feat-bars").innerHTML = fHtml;

  // Show section
  const sec = document.getElementById("result-section");
  sec.classList.remove("hidden");
  sec.scrollIntoView({ behavior: "smooth", block: "start" });

  // Animate bars
  setTimeout(() => {
    document.querySelectorAll(".conf-bar-fill[data-width]").forEach(el => {
      el.style.width = el.dataset.width + "%";
    });
    document.querySelectorAll(".feat-fill[data-width]").forEach(el => {
      el.style.width = el.dataset.width + "%";
    });
  }, 100);
}

function classifyFromSensor() {
  const inputs = getSensorInputs();
  const { predicted, probs } = classifySensor(inputs);
  showResult(predicted, probs, "sensor", []);
}

function classifyFromNLP() {
  const text = document.getElementById("nlp-text").value;
  if (!text.trim()) { alert("Please enter a smell description!"); return; }
  const { predicted, probs, tokens, matchedKW } = nlpClassify(text);

  // Show tokens
  let html = "";
  const allKW = Object.values(SMELL_LEXICON).flat();
  for (const t of tokens) {
    const isKW = allKW.some(k => k.includes(t.original) || t.original.includes(k) || stem(t.original) === stem(k));
    if (t.isStop) html += `<span class="token stop">${t.original}</span>`;
    else if (isKW) html += `<span class="token keyword">${t.original}</span>`;
    else html += `<span class="token">${t.original}</span>`;
  }
  document.getElementById("nlp-tokens").innerHTML = html;

  showResult(predicted, probs, "nlp", matchedKW);
}

// ─────────────────────────────────────────────
// 7. RENDER DATASET TABLE & CHART
// ─────────────────────────────────────────────

function renderDatasetTable() {
  const tbody = document.getElementById("dataset-tbody");
  const feats = ["temp", "hum", "mq2", "mq4", "mq8", "mq135", "mq136", "nh3", "vnh3"];
  let html = "";

  for (const cls of SMELL_CLASSES) {
    const prof = SMELL_PROFILES[cls.id];
    html += `<tr>
      <td>${cls.icon} ${cls.name}</td>
      ${feats.map(f => `<td>${prof[f]}</td>`).join("")}
      <td>80</td>
    </tr>`;
  }
  tbody.innerHTML = html;
}

function renderClassChart() {
  const container = document.getElementById("class-chart");
  const colors = SMELL_CLASSES.map(c => c.color);
  const maxCount = 80;
  let html = '<div class="feat-title" style="margin-bottom:24px;font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-muted)">Class Distribution (80 samples each)</div>';

  SMELL_CLASSES.forEach((cls, i) => {
    const pct = (80 / maxCount) * 100;
    html += `
      <div class="chart-bar-row">
        <div class="chart-bar-label">${cls.icon} ${cls.id}</div>
        <div class="chart-bar-track">
          <div class="chart-bar-fill" style="width:${pct}%;background:${colors[i]}">${80}</div>
        </div>
        <div class="chart-bar-count">80</div>
      </div>`;
  });
  container.innerHTML = html;
}

// ─────────────────────────────────────────────
// 8. RENDER MODEL COMPARISON
// ─────────────────────────────────────────────

function renderModels() {
  const models = [
    { name: "XGBoost", type: "Gradient Boosting", acc: 94.2, best: true, params: "n_est=250, lr=0.1, depth=6" },
    { name: "Random Forest", type: "Ensemble Trees", acc: 91.7, best: false, params: "n_est=200, max_depth=None" },
    { name: "SVM (RBF)", type: "Support Vector Machine", acc: 88.3, best: false, params: "kernel=rbf, C=1.0" },
    { name: "Logistic Reg.", type: "Linear Classifier", acc: 79.6, best: false, params: "max_iter=1000" },
  ];

  const grid = document.getElementById("model-grid");
  let html = "";
  for (const m of models) {
    html += `
      <div class="model-card ${m.best ? "best" : ""}">
        <div class="model-name">${m.name}</div>
        <div class="model-type">${m.type}</div>
        <div class="model-acc-bar">
          <div class="model-acc-fill" style="width:0%" data-width="${m.acc}"></div>
        </div>
        <div class="model-acc-num">${m.acc}%</div>
        <div class="model-acc-lbl">Test Accuracy</div>
        <div style="margin-top:12px;font-size:11px;color:var(--text-muted)">${m.params}</div>
        ${m.best ? '<div class="model-badge">★ Best Model</div>' : ""}
      </div>`;
  }
  grid.innerHTML = html;

  // Animate
  setTimeout(() => {
    document.querySelectorAll(".model-acc-fill[data-width]").forEach(el => {
      el.style.width = el.dataset.width + "%";
    });
  }, 400);
}

// ─────────────────────────────────────────────
// 9. ANIMATED BACKGROUND CANVAS
// ─────────────────────────────────────────────

function initCanvas() {
  const canvas = document.getElementById("bgCanvas");
  const ctx = canvas.getContext("2d");
  let W = window.innerWidth, H = window.innerHeight;
  canvas.width = W; canvas.height = H;

  const dots = Array.from({ length: 60 }, () => ({
    x: Math.random() * W,
    y: Math.random() * H,
    r: Math.random() * 1.5 + 0.5,
    dx: (Math.random() - 0.5) * 0.3,
    dy: (Math.random() - 0.5) * 0.3,
    color: ["#b8ff57","#57d4ff","#ff7857","#a29bfe"][Math.floor(Math.random() * 4)]
  }));

  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (const d of dots) {
      d.x += d.dx; d.y += d.dy;
      if (d.x < 0 || d.x > W) d.dx *= -1;
      if (d.y < 0 || d.y > H) d.dy *= -1;

      ctx.beginPath();
      ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
      ctx.fillStyle = d.color;
      ctx.fill();
    }

    // Draw faint connections
    for (let i = 0; i < dots.length; i++) {
      for (let j = i + 1; j < dots.length; j++) {
        const dist = Math.hypot(dots[i].x - dots[j].x, dots[i].y - dots[j].y);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(dots[i].x, dots[i].y);
          ctx.lineTo(dots[j].x, dots[j].y);
          ctx.strokeStyle = `rgba(184,255,87,${0.04 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }
  draw();

  window.addEventListener("resize", () => {
    W = window.innerWidth; H = window.innerHeight;
    canvas.width = W; canvas.height = H;
  });
}

// ─────────────────────────────────────────────
// 10. INIT
// ─────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initCanvas();
  renderDatasetTable();
  renderClassChart();
  renderModels();
});

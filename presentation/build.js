const pptxgen = require("pptxgenjs");
let pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
pres.author = "Sohaib";
pres.title = "Traffic Flow Prediction Using Deep Learning";

const BG = "0B1120";       // deep navy background
const CARD = "141C2E";     // card surface
const CYAN = "00D4FF";
const PURPLE = "7C3AED";
const GREEN = "22C55E";
const RED = "EF4444";
const WHITE = "F5F7FA";
const MUTED = "8B96AB";

const FIG = "../outputs/figures/";

function baseSlide(bg = BG) {
  let s = pres.addSlide();
  s.background = { color: bg };
  return s;
}

function header(s, kicker, title) {
  s.addText(kicker.toUpperCase(), { x: 0.6, y: 0.35, w: 8, h: 0.35, fontSize: 12, color: CYAN, bold: true, charSpacing: 2, fontFace: "Calibri" });
  s.addText(title, { x: 0.6, y: 0.68, w: 11.8, h: 0.7, fontSize: 28, color: WHITE, bold: true, fontFace: "Cambria" });
}

function pageNum(s, n) {
  s.addText(String(n), { x: 12.7, y: 7.05, w: 0.5, h: 0.35, fontSize: 10, color: MUTED, align: "right" });
}

// ---------- Slide 1: Title ----------
let s1 = baseSlide();
s1.addText("TRAFFIC FLOW PREDICTION", { x: 0.9, y: 2.35, w: 11.5, h: 1.0, fontSize: 44, bold: true, color: WHITE, fontFace: "Cambria" });
s1.addText("Using Deep Learning", { x: 0.9, y: 3.15, w: 11.5, h: 0.7, fontSize: 26, color: CYAN, fontFace: "Calibri" });
s1.addText("LSTM  vs  GRU  vs  Transformer  —  benchmarked on METR-LA (Los Angeles highway sensors)", {
  x: 0.9, y: 3.95, w: 11.5, h: 0.5, fontSize: 15, color: MUTED, fontFace: "Calibri"
});
s1.addShape(pres.shapes.OVAL, { x: 10.2, y: 0.9, w: 2.6, h: 2.6, fill: { color: PURPLE, transparency: 82 }, line: { type: "none" } });
s1.addShape(pres.shapes.OVAL, { x: 11.3, y: 4.6, w: 1.8, h: 1.8, fill: { color: CYAN, transparency: 85 }, line: { type: "none" } });
s1.addText("Sohaib  |  AI Automation Engineer  |  Deep Learning Project Presentation", {
  x: 0.9, y: 6.65, w: 11, h: 0.4, fontSize: 12, color: MUTED, fontFace: "Calibri"
});

// ---------- Slide 2: Problem & Dataset ----------
let s2 = baseSlide(WHITE);
header2(s2);
function header2(s) {
  s.addText("PROBLEM & DATASET", { x: 0.6, y: 0.35, w: 8, h: 0.35, fontSize: 12, color: PURPLE, bold: true, charSpacing: 2, fontFace: "Calibri" });
  s.addText("Forecasting Highway Speed From Sensor History", { x: 0.6, y: 0.68, w: 12, h: 0.7, fontSize: 28, color: "0B1120", bold: true, fontFace: "Cambria" });
}
const cardY = 1.85, cardH = 4.7;
const cards2 = [
  { t: "Dataset", body: "METR-LA — 207 real loop-detector sensors on LA highways, 5-minute readings, Mar–Jun 2012. This project uses a real 1,025-timestep slice (full 34k-row file needs Google-Drive access unavailable here)." , color: CYAN},
  { t: "Target", body: "Network-average speed (mph). Speed is the standard forecasting target in the METR-LA literature — congestion shows up as a speed drop through the fundamental traffic-flow relationship." , color: GREEN},
  { t: "Task", body: "Predict speed 5 minutes ahead from the last 60 minutes of history + calendar/lag/rolling features. Bonus: extend to 30- and 60-minute horizons." , color: PURPLE},
];
cards2.forEach((c, i) => {
  const x = 0.6 + i * 4.13;
  s2.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: cardY, w: 3.85, h: cardH, fill: { color: "FFFFFF" }, line: { color: "E2E8F0", width: 1 }, rectRadius: 0.08,
    shadow: { type: "outer", color: "0B1120", blur: 8, offset: 3, angle: 90, opacity: 0.08 } });
  s2.addShape(pres.shapes.OVAL, { x: x + 0.3, y: cardY + 0.3, w: 0.5, h: 0.5, fill: { color: c.color } });
  s2.addText(c.t, { x: x + 0.3, y: cardY + 1.0, w: 3.3, h: 0.4, fontSize: 17, bold: true, color: "0B1120", fontFace: "Cambria" });
  s2.addText(c.body, { x: x + 0.3, y: cardY + 1.45, w: 3.3, h: 3.1, fontSize: 13, color: "334155", fontFace: "Calibri", valign: "top" });
});

// ---------- Slide 3: EDA ----------
let s3 = baseSlide(WHITE);
s3.addText("EXPLORATORY DATA ANALYSIS", { x: 0.6, y: 0.35, w: 8, h: 0.35, fontSize: 12, color: PURPLE, bold: true, charSpacing: 2, fontFace: "Calibri" });
s3.addText("Clear Diurnal Pattern, Sentinel Missing Values", { x: 0.6, y: 0.68, w: 12, h: 0.7, fontSize: 28, color: "0B1120", bold: true, fontFace: "Cambria" });
s3.addImage({ path: FIG + "03_hourly_pattern.png", x: 0.7, y: 1.8, w: 8.2, h: 2.73 });
const bullets3 = [
  "0.0 mph readings (~2.8%) are a sensor-dropout sentinel, not real data",
  "Speeds cluster 55–70 mph (free flow), heavier low-speed tail = congestion",
  "AM/PM commute dips are visible \u2192 motivates hour-of-day features",
  "Nearby sensors are highly correlated (spatial redundancy)",
];
s3.addText(bullets3.map(b => ({ text: b, options: { bullet: { code: "2192" }, breakLine: true, color: "0B1120" } })),
  { x: 9.2, y: 2.0, w: 3.5, h: 3.2, fontSize: 13.5, fontFace: "Calibri", valign: "top" });

// ---------- Slide 4: Preprocessing & Feature Engineering ----------
let s4 = baseSlide(BG);
header(s4, "Methodology", "Preprocessing & Feature Engineering");
const stepsL = [
  ["Missing values", "0 \u2192 NaN \u2192 time-aware interpolation (30-min limit) \u2192 edge-fill"],
  ["Outliers", "Domain-based clip to [0, 80] mph"],
  ["Calendar features", "hour_sin / hour_cos (cyclical), is_weekend"],
];
const stepsR = [
  ["Lag features", "t-1, t-2, t-3, t-6, t-12 (5\u201360 min back)"],
  ["Rolling stats", "30-min & 60-min rolling mean/std (shift(1), no leakage)"],
  ["Sequencing", "60-min lookback (12 steps) \u2192 5-min-ahead target"],
];
function stepCol(s, items, x) {
  items.forEach((it, i) => {
    const y = 2.0 + i * 1.55;
    s.addShape(pres.shapes.OVAL, { x, y, w: 0.45, h: 0.45, fill: { color: CYAN } });
    s.addText(String(i + 1), { x, y, w: 0.45, h: 0.45, fontSize: 16, bold: true, color: BG, align: "center", valign: "middle" });
    s.addText(it[0], { x: x + 0.65, y: y - 0.05, w: 5, h: 0.4, fontSize: 16, bold: true, color: WHITE, fontFace: "Cambria" });
    s.addText(it[1], { x: x + 0.65, y: y + 0.35, w: 5, h: 0.8, fontSize: 12.5, color: MUTED, fontFace: "Calibri" });
  });
}
stepCol(s4, stepsL, 0.7);
stepCol(s4, stepsR, 6.9);
pageNum(s4, 4);

// ---------- Slide 5: Model Architectures ----------
let s5 = baseSlide(WHITE);
s5.addText("MODELS", { x: 0.6, y: 0.35, w: 8, h: 0.35, fontSize: 12, color: PURPLE, bold: true, charSpacing: 2, fontFace: "Calibri" });
s5.addText("Three Sequence Architectures, Same Inputs & Splits", { x: 0.6, y: 0.68, w: 12, h: 0.7, fontSize: 28, color: "0B1120", bold: true, fontFace: "Cambria" });
const models5 = [
  { t: "LSTM", d: "3-gate recurrent unit (input, forget, output). Strong at long-range dependencies.", c: RED },
  { t: "GRU", d: "2-gate recurrent unit (update, reset). Fewer parameters \u2192 less overfitting on small data.", c: GREEN },
  { t: "Transformer", d: "Self-attention + positional encoding. No recurrent bias \u2014 needs more data to shine.", c: CYAN },
];
models5.forEach((m, i) => {
  const x = 0.6 + i * 4.13;
  s5.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.85, w: 3.85, h: 4.7, fill: { color: "FFFFFF" }, line: { color: "E2E8F0", width: 1 }, rectRadius: 0.08,
    shadow: { type: "outer", color: "0B1120", blur: 8, offset: 3, angle: 90, opacity: 0.08 } });
  s5.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.3, y: 2.15, w: 3.25, h: 0.7, fill: { color: m.c, transparency: 88 }, line: { type: "none" }, rectRadius: 0.06 });
  s5.addText(m.t, { x: x + 0.3, y: 2.15, w: 3.25, h: 0.7, fontSize: 20, bold: true, color: "0B1120", align: "center", valign: "middle", fontFace: "Cambria" });
  s5.addText(m.d, { x: x + 0.3, y: 3.1, w: 3.3, h: 1.6, fontSize: 13, color: "334155", fontFace: "Calibri", valign: "top" });
  s5.addText("Linear head \u2192 speed (mph)", { x: x + 0.3, y: 5.9, w: 3.3, h: 0.5, fontSize: 11.5, italic: true, color: MUTED, fontFace: "Calibri" });
});

// ---------- Slide 6: Hyperparameter Tuning ----------
let s6 = baseSlide(BG);
header(s6, "Methodology", "Hyperparameter Tuning (LSTM & GRU)");
s6.addText("Narrow grid search \u2014 hidden_size \u00d7 num_layers, selected by validation MSE. Kept deliberately small: with ~700 training sequences, an exhaustive search tunes to validation noise more than it informs architecture choice.",
  { x: 0.7, y: 1.85, w: 11.8, h: 0.9, fontSize: 14, color: MUTED, fontFace: "Calibri" });
const rows6 = [
  ["Model", "hidden_size", "num_layers", "Val MSE (best)"],
  ["LSTM", "32", "1", "0.00151"],
  ["GRU", "32", "1", "0.00105  \u2b50 best overall"],
];
s6.addTable(rows6, {
  x: 0.7, y: 2.9, w: 8.5, h: 1.8, fontSize: 14, color: WHITE, fontFace: "Calibri",
  border: { type: "solid", color: "223047", pt: 1 },
  fill: { color: CARD },
  colW: [2.2, 2.2, 2.2, 1.9],
  autoPage: false,
});
s6.addImage({ path: FIG + "05_val_loss_curves.png", x: 0.7, y: 5.1, w: 8.5, h: 1.9 });
pageNum(s6, 6);

// ---------- Slide 7: Results ----------
let s7 = baseSlide(WHITE);
s7.addText("RESULTS", { x: 0.6, y: 0.35, w: 8, h: 0.35, fontSize: 12, color: PURPLE, bold: true, charSpacing: 2, fontFace: "Calibri" });
s7.addText("GRU Wins Clearly on the Held-out Test Set", { x: 0.6, y: 0.68, w: 12, h: 0.7, fontSize: 28, color: "0B1120", bold: true, fontFace: "Cambria" });
const rows7 = [
  ["Model", "MAE (mph)", "RMSE (mph)", "MAPE (%)", "R\u00b2"],
  ["GRU", "0.34", "0.45", "0.54", "0.90"],
  ["LSTM", "0.64", "0.78", "0.99", "0.69"],
  ["Transformer", "0.78", "0.96", "1.21", "0.53"],
];
s7.addTable(rows7, {
  x: 0.7, y: 1.85, w: 11.7, h: 1.9, fontSize: 15, color: "0B1120", fontFace: "Calibri",
  border: { type: "solid", color: "E2E8F0", pt: 1 },
  fill: { color: "F8FAFC" },
  colW: [3.5, 2.05, 2.05, 2.05, 2.05],
  autoPage: false,
});
s7.addImage({ path: FIG + "06_metrics_comparison.png", x: 0.5, y: 4.0, w: 12.3, h: 2.73 });

// ---------- Slide 8: Actual vs Predicted + Why ----------
let s8 = baseSlide(WHITE);
s8.addText("RESULTS", { x: 0.6, y: 0.35, w: 8, h: 0.35, fontSize: 12, color: PURPLE, bold: true, charSpacing: 2, fontFace: "Calibri" });
s8.addText("Why GRU Wins Here", { x: 0.6, y: 0.68, w: 12, h: 0.7, fontSize: 28, color: "0B1120", bold: true, fontFace: "Cambria" });
s8.addImage({ path: FIG + "07_actual_vs_predicted.png", x: 0.6, y: 1.75, w: 6.2, h: 4.43 });
const why8 = [
  "GRU has ~25% fewer parameters than LSTM at the same hidden size \u2192 lower-variance fit on only ~700 training sequences",
  "LSTM's extra cell-state gate is built for long dependencies \u2014 barely used by a 60-min lookback, so it mostly adds noise here",
  "Transformer has no recurrent inductive bias \u2014 needs far more data to learn temporal locality from scratch",
  "Expect this ranking to shift toward Transformer on the full 34k-row dataset with a graph-aware spatial component",
];
s8.addText(why8.map(b => ({ text: b, options: { bullet: { code: "2192" }, breakLine: true, color: "0B1120" } })),
  { x: 7.1, y: 1.9, w: 5.6, h: 4.3, fontSize: 13.5, fontFace: "Calibri", valign: "top" });

// ---------- Slide 9: Conclusion & Future Work ----------
let s9 = baseSlide(BG);
s9.addText("CONCLUSION", { x: 0.6, y: 0.35, w: 8, h: 0.35, fontSize: 12, color: CYAN, bold: true, charSpacing: 2, fontFace: "Calibri" });
s9.addText("Takeaways & Where This Goes Next", { x: 0.6, y: 0.68, w: 12, h: 0.7, fontSize: 28, color: WHITE, bold: true, fontFace: "Cambria" });
s9.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.85, w: 5.9, h: 4.7, fill: { color: CARD }, line: { type: "none" }, rectRadius: 0.08 });
s9.addText("Bottom line", { x: 1.0, y: 2.15, w: 5.2, h: 0.4, fontSize: 17, bold: true, color: GREEN, fontFace: "Cambria" });
s9.addText("On limited data, GRU is the pragmatic choice for short-horizon forecasting. Transformers need volume \u2014 and ideally spatial structure \u2014 to earn their flexibility.",
  { x: 1.0, y: 2.65, w: 5.2, h: 1.6, fontSize: 13.5, color: MUTED, fontFace: "Calibri" });
s9.addText("Bonus delivered", { x: 1.0, y: 4.3, w: 5.2, h: 0.4, fontSize: 17, bold: true, color: CYAN, fontFace: "Cambria" });
s9.addText("30-min and 60-min direct multi-step GRU forecasts, with error growing predictably with horizon length.",
  { x: 1.0, y: 4.8, w: 5.2, h: 1.4, fontSize: 13.5, color: MUTED, fontFace: "Calibri" });

s9.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.8, y: 1.85, w: 5.9, h: 4.7, fill: { color: CARD }, line: { type: "none" }, rectRadius: 0.08 });
s9.addText("Future work", { x: 7.2, y: 2.15, w: 5.2, h: 0.4, fontSize: 17, bold: true, color: PURPLE, fontFace: "Cambria" });
const future9 = [
  "Train on the full 34,272-row METR-LA release",
  "Graph neural network (DCRNN / Graph WaveNet) to model all 207 sensors jointly",
  "Recursive/seq2seq decoder for longer multi-step horizons",
  "Wrap saved models in a live Streamlit dashboard",
];
s9.addText(future9.map(b => ({ text: b, options: { bullet: true, breakLine: true, color: WHITE } })),
  { x: 7.2, y: 2.65, w: 5.3, h: 3.6, fontSize: 13.5, fontFace: "Calibri", valign: "top" });

pres.writeFile({ fileName: "Traffic_Flow_Prediction_Presentation.pptx" }).then(() => console.log("done"));

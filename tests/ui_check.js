const { JSDOM } = require("jsdom");
const sleep = ms => new Promise(r => setTimeout(r, ms));
const fails = [];
const check = (c, m) => { if (!c) { fails.push(m); console.log("FAIL:", m); } else console.log("ok  :", m); };
(async () => {
  const dom = await JSDOM.fromURL((process.env.NEUROOPT_URL || "http://127.0.0.1:8765/"), {
    runScripts: "dangerously", resources: "usable", pretendToBeVisual: true,
    beforeParse(w) { w.fetch = (u, o) => fetch(new URL(u, process.env.NEUROOPT_URL || "http://127.0.0.1:8765/"), o); }
  });
  const w = dom.window, d = w.document;
  const errors = []; w.addEventListener("error", e => errors.push(e.message));
  for (let i = 0; i < 50 && d.querySelectorAll("#example option").length < 7; i++) await sleep(100);
  check(d.querySelectorAll("#example option").length === 7, "examples dropdown populated (6 + placeholder)");
  check(d.querySelectorAll("#oracle option").length >= 1, "oracle dropdown populated");
  check(d.querySelector("#asm").value.includes("imul rbx, rbx, 8"), "worked example preloaded");
  check(d.querySelector("#measure").classList.contains("hidden"), "timing button hidden when disabled");

  d.querySelector("#run").click();
  for (let i = 0; i < 80 && d.querySelector("#results").classList.contains("hidden"); i++) await sleep(100);
  check(!d.querySelector("#results").classList.contains("hidden"), "results shown after Optimize");
  const txt = s => d.querySelector(s).textContent;
  check(txt("#cb") === "6.00" && txt("#ca") === "2.25", "cost 6.00 -> 2.25 displayed");
  check(txt("#redtxt").includes("62.5%"), "62.5% reduction displayed");
  check(txt("#ob") === "17" && txt("#nb") === "10", "bytes 17 -> 10 displayed");
  check(txt("#verdict").includes("EQUIVALENT"), "emulator verdict EQUIVALENT shown");
  check(d.querySelectorAll("#decode tbody tr").length === 4, "4 decoded instruction rows");
  check(d.querySelectorAll("#decode tbody tr:last-child .pill.on").length === 6, "last row shows all six flags live");
  check(d.querySelectorAll("#decode tbody tr:first-child .pill.on").length === 0, "first row shows no live flags");
  check(d.querySelectorAll("#legal tbody tr").length === 3, "3 legal rewrites listed");
  check(txt("#steps").includes("R2 @1") && txt("#steps").includes("R1 @0"), "selection steps listed");
  check(txt("#proxynote").includes("not a hardware speedup"), "proxy-not-speedup note shown");
  check(d.querySelectorAll("#compare tbody tr").length >= 1, "oracle comparison table filled");

  // trap example
  const sel = d.querySelector("#example"); sel.value = "1"; sel.dispatchEvent(new w.Event("change"));
  check(d.querySelector("#asm").value.includes("adc rax, rbx"), "selecting trap example fills the box");
  d.querySelector("#run").click(); await sleep(1500);
  check(txt("#refused").includes("CF read by adc"), "adc trap shows the refusal reason");
  check(txt("#steps").includes("original block is kept"), "no rewrite accepted for the trap");

  // error path
  d.querySelector("#asm").value = "frobnicate rax"; d.querySelector("#run").click(); await sleep(1500);
  check(!d.querySelector("#error").classList.contains("hidden") && txt("#error").includes("Could not assemble"), "assembler error shown in the page");
  check(d.querySelector("#results").classList.contains("hidden"), "results hidden on error");

  // user text must never be injected as HTML
  d.querySelector("#asm").value = "mov rax, 0"; d.querySelector("#hex").value = "<img src=x onerror=alert(1)>"; d.querySelector("#run").click(); await sleep(1500);
  check(d.querySelector("#error img") === null, "no HTML injection through error text");
  check(errors.length === 0, "no JavaScript errors (" + errors.join("; ") + ")");
  console.log(fails.length ? "\n" + fails.length + " FAILED" : "\nALL UI CHECKS PASSED");
  process.exit(fails.length ? 1 : 0);
})();

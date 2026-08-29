/* Transformer Benchmark Dashboard - client
 * ============================================================================
 *
 * Vanilla JS, one file, no build step, matching a repository whose only
 * dependencies are torch and ninja.
 *
 * The shape of it: /api/spec is fetched once and describes every form field, so
 * nothing here hard-codes the harness's flags. A form is a plain object keyed by
 * argparse dest, and a field whose value equals what is already in force is
 * deleted from it rather than stored - which is what keeps the previewed command
 * as short as one typed by hand.
 *
 * Results are rendered twice on purpose: a summary that leads with the single
 * number the view is about, and the full table underneath. Every value in the
 * chart is also in the table, so nothing is reachable only by hovering.
 */
'use strict';

/* ----------------------------------------------------------------- theme -- */

/* Three states. "system" stamps nothing and lets prefers-color-scheme decide;
 * light/dark stamp data-theme, which the stylesheet gives precedence over the
 * media query in both directions. */
const THEME_KEY = 'tb-theme';

function applyTheme(choice) {
  if (choice === 'light' || choice === 'dark') {
    document.documentElement.setAttribute('data-theme', choice);
  } else {
    document.documentElement.removeAttribute('data-theme');
  }
  document.querySelectorAll('[data-theme-choice]').forEach((button) => {
    button.setAttribute('aria-pressed',
      String(button.dataset.themeChoice === (choice || 'system')));
  });
  try { localStorage.setItem(THEME_KEY, choice || 'system'); } catch (e) { /* private mode */ }
}

// Applied before first paint so the page does not flash the wrong theme.
try { applyTheme(localStorage.getItem(THEME_KEY) || 'system'); } catch (e) { /* ignore */ }

/* ------------------------------------------------------------- dom utils -- */

const $ = (id) => document.getElementById(id);

const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

const clear = (node) => { while (node && node.firstChild) node.removeChild(node.firstChild); };

/* "optimized/config.py: ATTENTION_IMPL" -> "config.py". Enough to say where to
 * look, short enough to sit inside a dropdown. The harness's own filename is
 * long and says nothing a reader here does not already know. */
function shortSource(source) {
  if (!source) return 'the harness';
  const file = String(source).split(':')[0].trim();
  if (file.indexOf('torch_transformer_benchmark') !== -1) return 'the harness';
  return file.split('/').pop() || file;
}

const VIEW_TITLES = {
  run: ['Run', 'One configuration on one shape'],
  compare: ['Compare', 'Two configurations, timed back to back against a control'],
  sweep: ['Sweep', 'One configuration across many shapes'],
  scripts: ['Scripts', "The repository's own A/B and verification scripts"],
  presets: ['Presets', 'The saved shapes, written to presets.json'],
  history: ['History', 'Finished jobs and where their logs are'],
};

const state = {
  spec: null,
  forms: { run: {}, cmpShape: {}, cmpA: {}, cmpB: {}, sweep: {}, script: {} },
  jobs: { run: null, compare: null, sweep: null, script: null },
  polls: {},
  selectedScript: null,
};

// Filled in by setupSweep so a preset save can rebuild the shape list.
let rebuildShapeList = null;

/* ------------------------------------------------------------------- api -- */

async function api(path, options) {
  const response = await fetch(path, options);
  let payload = null;
  try { payload = await response.json(); } catch (e) { payload = null; }
  return { ok: response.ok, status: response.status, data: payload };
}

const postJSON = (path, body) => api(path, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body || {}),
});

/* -------------------------------------------------------------- controls -- */

/* One argparse flag -> one labelled input bound to `form`.
 *
 * Every control opens on the value actually in force. For the five optimization
 * flags argparse reports default=None, because the real value lives in
 * optimized/config.py and cli.py only overrides it when the flag is passed;
 * argspec digs that out so the dropdown can preselect it. Picking the
 * preselected value back does not lengthen the command - build_argv drops any
 * value equal to the effective one.
 *
 * The per-mode explanations config.py already carries beside each constant
 * become the tooltip on each option, and are shown in full under the field for
 * whichever mode is selected - including "auto", whose name says nothing about
 * what it does. */
function makeField(spec, form, onChange) {
  const wrap = el('label', 'field');
  if (spec.help) wrap.title = spec.help;

  const label = el('span', 'field-label', spec.flag.replace(/^--/, ''));
  let input;

  // "Set" means the field will actually change the run: present, and different
  // from what is already in force. Selecting the preselected mode back out of a
  // dropdown is not a change, and marking it as one would make every form look
  // edited the moment it loaded.
  const isUntouched = () => {
    const chosen = form[spec.dest];
    if (chosen === undefined) return true;
    if (spec.effective === null || spec.effective === undefined) return false;
    return String(chosen) === String(spec.effective);
  };
  const mark = () => wrap.classList.toggle('is-set', !isUntouched());

  if (spec.kind === 'flag') {
    wrap.classList.add('is-check');
    input = el('input');
    input.type = 'checkbox';
    input.checked = !!form[spec.dest];
    wrap.appendChild(input);
    wrap.appendChild(label);
    input.addEventListener('change', () => {
      if (input.checked) form[spec.dest] = true; else delete form[spec.dest];
      mark(); onChange();
    });
  } else if (spec.kind === 'choice' || spec.kind === 'tristate') {
    wrap.appendChild(label);
    input = el('select');
    const help = spec.choice_help || {};
    // A tristate's value arrives as a JSON boolean, and "True" beside options
    // labelled on/off reads as a third state that does not exist.
    const effective = spec.effective === null || spec.effective === undefined
      ? null
      : (spec.kind === 'tristate' ? (spec.effective ? 'on' : 'off') : String(spec.effective));
    const effectiveValue = spec.kind === 'tristate'
      ? (spec.effective ? 'true' : 'false') : effective;

    const options = spec.kind === 'tristate'
      ? [['true', 'on'], ['false', 'off']]
      : (spec.choices || []).map((c) => [c, c]);

    options.forEach(([value, text]) => {
      const option = new Option(text, value);
      if (help[value]) option.title = help[value];
      input.appendChild(option);
    });
    input.value = form[spec.dest] === undefined
      ? (effectiveValue === null ? '' : String(effectiveValue))
      : String(form[spec.dest]);
    wrap.appendChild(input);

    const note = el('span', 'note');
    const showNote = () => {
      const active = input.value || effectiveValue;
      const text = help[active];
      note.textContent = text ? active + ': ' + text : '';
      note.hidden = !text;
    };
    wrap.appendChild(note);
    showNote();

    input.addEventListener('change', () => {
      if (input.value === '') delete form[spec.dest];
      else if (spec.kind === 'tristate') form[spec.dest] = input.value === 'true';
      else form[spec.dest] = input.value;
      showNote(); mark(); onChange();
    });
  } else {
    wrap.appendChild(label);
    input = el('input');
    input.type = (spec.kind === 'int' || spec.kind === 'float') ? 'number' : 'text';
    if (spec.kind === 'float') input.step = 'any';
    // The placeholder is the value the run uses if this is left empty, so say
    // where it comes from rather than leaving a bare grey number to be guessed.
    if (spec.effective !== null && spec.effective !== undefined) {
      input.placeholder = String(spec.effective);
      wrap.title = (spec.help ? spec.help + '\n\n' : '')
        + 'Left empty this stays ' + spec.effective + ', as set in '
        + shortSource(spec.effective_source) + '.';
    }
    if (form[spec.dest] !== undefined) input.value = form[spec.dest];
    wrap.appendChild(input);
    input.addEventListener('input', () => {
      const raw = input.value.trim();
      if (raw === '') delete form[spec.dest];
      else if (spec.kind === 'int') form[spec.dest] = parseInt(raw, 10);
      else if (spec.kind === 'float') form[spec.dest] = parseFloat(raw);
      else form[spec.dest] = raw;
      mark(); onChange();
    });
  }

  input.dataset.dest = spec.dest;
  wrap.dataset.dest = spec.dest;
  mark();
  return wrap;
}

function renderGroup(container, group, form, onChange) {
  if (!container) return;
  clear(container);
  state.spec.fields
    .filter((spec) => spec.group === group)
    .forEach((spec) => container.appendChild(makeField(spec, form, onChange)));
}

/* The environment-variable knobs. No "leave alone" option is needed: their
 * in-force value is a real one the control already shows. The variable name
 * leads the tooltip because that is what appears in the previewed command. */
function renderEnv(container, form, onChange) {
  if (!container) return;
  clear(container);
  if (!form.env) form.env = {};

  state.spec.env_knobs.forEach((knob) => {
    const wrap = el('label', 'field');
    const inForce = knob.kind === 'bool' ? (knob.default ? 'on' : 'off') : knob.default;
    wrap.title = knob.name + '\n\n' + knob.help + '\n\nCurrently ' + inForce
      + '; read in ' + knob.source + '.';
    const label = el('span', 'field-label', knob.label || knob.name);
    let input;

    if (knob.kind === 'bool') {
      wrap.classList.add('is-check');
      input = el('input');
      input.type = 'checkbox';
      input.checked = form.env[knob.name] === undefined ? knob.default : !!form.env[knob.name];
      wrap.appendChild(input);
      wrap.appendChild(label);
      input.addEventListener('change', () => {
        if (input.checked === knob.default) delete form.env[knob.name];
        else form.env[knob.name] = input.checked;
        wrap.classList.toggle('is-set', form.env[knob.name] !== undefined);
        onChange();
      });
    } else {
      wrap.appendChild(label);
      input = el('input');
      input.type = 'number';
      input.placeholder = String(knob.default);
      if (form.env[knob.name] !== undefined) input.value = form.env[knob.name];
      wrap.appendChild(input);
      input.addEventListener('input', () => {
        const raw = input.value.trim();
        const value = parseInt(raw, 10);
        if (raw === '' || value === knob.default || Number.isNaN(value)) delete form.env[knob.name];
        else form.env[knob.name] = value;
        wrap.classList.toggle('is-set', form.env[knob.name] !== undefined);
        onChange();
      });
    }
    // Needed by setFieldValue, which is how "copy A -> B" reaches these.
    input.dataset.dest = knob.name;
    wrap.dataset.dest = knob.name;
    wrap.classList.toggle('is-set', form.env[knob.name] !== undefined);
    container.appendChild(wrap);
  });
}

function setFieldValue(container, dest, value) {
  const holder = container.querySelector('[data-dest="' + dest + '"]');
  if (!holder) return;
  const node = holder.tagName === 'LABEL' ? holder.querySelector('input,select') : holder;
  if (!node) return;
  if (node.type === 'checkbox') node.checked = !!value;
  else node.value = value === undefined || value === null ? '' : String(value);
  node.dispatchEvent(new Event(node.tagName === 'SELECT' || node.type === 'checkbox'
    ? 'change' : 'input'));
}

/* ------------------------------------------------------------- preflight -- */

function escapeHtml(text) {
  return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function highlightCommand(text) {
  return escapeHtml(text)
    .replace(/(^|\s)([A-Z][A-Z0-9_]{3,})=(\S+)/g, '$1<span class="tok-env">$2=$3</span>')
    .replace(/(\s)(--[a-z0-9-]+)/g, '$1<span class="tok-flag">$2</span>');
}

function renderIssues(container, issues) {
  if (!container) return;
  clear(container);
  (issues || []).forEach((issue) => {
    const node = el('div', 'issue' + (issue.level === 'error' ? ' is-error' : ''));
    // The glyph is what keeps a status colour from carrying the meaning alone.
    node.appendChild(el('span', 'glyph', issue.level === 'error' ? '✖' : '⚠'));
    const body = el('span');
    body.appendChild(el('b', null, issue.level === 'error' ? 'Blocked. ' : 'Note. '));
    body.appendChild(document.createTextNode(issue.message));
    node.appendChild(body);
    container.appendChild(node);
  });
}

function renderDerived(container, estimate) {
  if (!container || !estimate) return;
  clear(container);
  const add = (label, value, title) => {
    const span = el('span');
    if (title) span.title = title;
    span.appendChild(el('b', null, label + ' '));
    span.appendChild(document.createTextNode(value));
    container.appendChild(span);
  };
  const plan = estimate.plan || {};
  add('head_dim', estimate.head_dim === null ? 'not divisible' : estimate.head_dim,
    'd_model / heads. Decides which attention kernel can run the shape: wmma and '
    + 'scalar cover 8-128, the tile kernels 8-64.');
  add('tokens', estimate.tokens.toLocaleString());
  add('input', estimate.human.input);
  add('predicted peak', estimate.human.peak,
    'What the harness predicts one forward needs: ten times a single '
    + '[rows, seq, d_model] activation, measured across the appendix shapes.');
  if (plan.slices > 1) {
    add('streamed', plan.slices + ' slices of ' + plan.rows,
      'The full input does not fit, so the harness runs the batch in slices. Rows '
      + 'of a batch do not interact, so this is the same computation - but the '
      + 'reported median is per slice.');
  }
  if (plan.known && plan.baseline_runs === false) {
    add('baseline', 'skipped',
      'The baseline cannot hold its score matrix at this shape, so the harness '
      + 'times the optimized model alone: latencies but no speedup.');
  }
}

/* Debounced. The server owns the rules, so the client asks rather than
 * reimplementing them and drifting. */
function makePreflight(getForm, commandNode, issuesNode, derivedNode, onResult) {
  let timer = null;
  return function schedule() {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const { data } = await postJSON('/api/preflight', { form: getForm() });
      if (!data) return;
      if (commandNode) commandNode.innerHTML = highlightCommand(data.command);
      if (issuesNode) renderIssues(issuesNode, data.issues);
      if (derivedNode) renderDerived(derivedNode, data.estimate);
      if (onResult) onResult(data);
    }, 220);
  };
}

/* --------------------------------------------------------------- results -- */

const fmt = (value, digits) =>
  (value === null || value === undefined) ? '–' : Number(value).toFixed(digits);

function rowFromStep(step) {
  const result = step.result || {};
  const optimized = result.optimized || {};
  const baseline = result.baseline || {};
  return {
    label: step.label,
    role: (step.meta && step.meta.role) || '',
    status: step.status,
    exitCode: step.exit_code,
    accuracy: result.accuracy_passed === null || result.accuracy_passed === undefined
      ? null : result.accuracy_passed,
    maxAbs: result.max_abs_error,
    maxRel: result.max_relative_error,
    failed: result.failed_elements === null || result.failed_elements === undefined
      ? null : result.failed_elements + '/' + result.total_elements,
    baseline: baseline.median_ms,
    optimized: optimized.median_ms,
    p90: optimized.p90_ms,
    min: optimized.min_ms,
    speedup: result.speedup,
    throughput: optimized.tokens_per_second,
    secs: step.duration_s,
    accuracySkipped: result.benchmark_skipped,
    baselineSkipped: result.baseline_skipped,
    totalMs: result.total_ms,
    slices: result.slices,
    rowsPerSlice: result.rows_per_slice,
    errors: result.error_lines || [],
  };
}

const COLUMNS = [
  { key: 'label', title: 'run', left: true },
  { key: 'status', title: 'state' },
  { key: 'accuracy', title: 'accuracy' },
  { key: 'maxAbs', title: 'max_abs' },
  { key: 'maxRel', title: 'max_rel' },
  { key: 'failed', title: 'failed' },
  { key: 'baseline', title: 'baseline ms' },
  { key: 'optimized', title: 'optimized ms' },
  { key: 'p90', title: 'p90 ms' },
  { key: 'min', title: 'min ms' },
  { key: 'speedup', title: 'speedup' },
  { key: 'throughput', title: 'token/s' },
  { key: 'secs', title: 'took' },
];

function fillCell(cell, key, row) {
  switch (key) {
    case 'label': {
      if (row.role) {
        cell.appendChild(el('span', 'role is-' + row.role.toLowerCase(),
          row.role === 'control' ? 'C' : row.role));
      }
      cell.appendChild(document.createTextNode(row.label));
      if (row.errors && row.errors.length) cell.title = row.errors.join('\n');
      break;
    }
    case 'status': {
      cell.textContent = row.status
        + (row.exitCode !== null && row.exitCode !== undefined && row.exitCode !== 0
          ? ' (' + row.exitCode + ')' : '');
      if (row.status === 'failed') cell.classList.add('is-critical');
      else if (row.status === 'done') cell.classList.add('is-good');
      else cell.classList.add('is-muted');
      break;
    }
    case 'accuracy': {
      if (row.accuracy === null) {
        cell.textContent = row.baselineSkipped ? 'n/a' : '–';
        cell.classList.add('is-muted');
        if (row.baselineSkipped) {
          cell.title = 'No baseline ran, so there is no reference to check against.';
        }
      } else {
        cell.textContent = row.accuracy ? 'PASS' : 'FAIL';
        cell.classList.add(row.accuracy ? 'is-good' : 'is-critical');
      }
      break;
    }
    case 'maxAbs':
    case 'maxRel': {
      const value = key === 'maxAbs' ? row.maxAbs : row.maxRel;
      cell.textContent = (value === null || value === undefined)
        ? '–' : Number(value).toExponential(2);
      break;
    }
    case 'failed': cell.textContent = row.failed === null ? '–' : row.failed; break;
    case 'baseline': {
      if (row.baseline === null || row.baseline === undefined) {
        cell.textContent = row.baselineSkipped ? 'cannot run' : '–';
        cell.classList.add('is-muted');
        if (row.baselineSkipped) {
          cell.title = 'The baseline cannot hold its score matrix at this shape, so '
            + 'the harness skipped it and timed the optimized model alone.';
        }
      } else cell.textContent = fmt(row.baseline, 4);
      break;
    }
    case 'optimized': {
      cell.textContent = fmt(row.optimized, 4);
      // On a streamed shape the median is one slice; a whole forward pass is the
      // sum over slices, and showing only the per-slice figure would understate
      // the run by that factor.
      if (row.slices > 1) {
        cell.textContent += '/slice';
        if (row.totalMs) cell.textContent += ' (' + fmt(row.totalMs, 0) + ' total)';
        cell.title = row.slices + ' slices of ' + row.rowsPerSlice + ' row(s); the '
          + 'median is per slice, the total is one full pass over the batch.';
      }
      break;
    }
    case 'p90': cell.textContent = fmt(row.p90, 4); break;
    case 'min': cell.textContent = fmt(row.min, 4); break;
    case 'speedup': {
      if (row.speedup === null || row.speedup === undefined) {
        cell.textContent = row.accuracySkipped ? 'accuracy failed'
          : (row.baselineSkipped ? 'no baseline' : '–');
        cell.classList.add('is-muted');
      } else {
        cell.textContent = row.speedup.toFixed(3) + 'x';
        cell.classList.add('is-strong', row.speedup >= 1 ? 'is-good' : 'is-critical');
      }
      break;
    }
    case 'throughput': {
      cell.textContent = (row.throughput === null || row.throughput === undefined)
        ? '–' : Math.round(row.throughput).toLocaleString();
      break;
    }
    case 'secs': {
      cell.textContent = (row.secs === null || row.secs === undefined)
        ? '–' : row.secs + 's';
      break;
    }
    default: cell.textContent = '–';
  }
}

function renderTable(table, rows, extraColumns) {
  const columns = COLUMNS.concat(extraColumns || []);
  clear(table);

  const head = table.createTHead().insertRow();
  columns.forEach((column) => head.appendChild(el('th', column.left ? 'left' : null, column.title)));

  const body = table.createTBody();
  rows.forEach((row) => {
    const tr = body.insertRow();
    if (row.role === 'control') tr.className = 'is-context';
    columns.forEach((column) => {
      const cell = tr.insertCell();
      if (column.left) cell.className = 'left';
      if (column.render) column.render(cell, row);
      else fillCell(cell, column.key, row);
    });
  });
}

function tableToCSV(table) {
  return Array.from(table.rows).map((row) =>
    Array.from(row.cells).map((cell) => {
      const text = cell.textContent.replace(/"/g, '""');
      return /[",\n]/.test(text) ? '"' + text + '"' : text;
    }).join(',')).join('\n');
}

/* --- the summary: hero, tiles, and the speedup bars ------------------------ */

function geomean(values) {
  if (!values.length) return null;
  return Math.exp(values.reduce((sum, v) => sum + Math.log(v), 0) / values.length);
}

function niceTicks(max) {
  // 4-5 ticks on a round step, so the axis reads without arithmetic.
  const raw = max / 4;
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude)
    .find((candidate) => candidate >= raw) || magnitude * 10;
  const ticks = [];
  for (let value = 0; value <= max + 1e-9; value += step) ticks.push(value);
  return ticks;
}

/* Speedup by shape.
 *
 * Bar length carries the magnitude, so the fill is ONE hue - a second channel
 * repeating the same thing is noise. A bar below 1.0x is a different kind of
 * result rather than a smaller one, so it takes the critical status colour and
 * is called "slower" in the legend. The 1.0x parity rule is solid, not dashed:
 * dashes read as projection, and this is a real reference. */
function renderBars(container, rows) {
  const scored = rows.filter((r) => typeof r.speedup === 'number' && r.speedup > 0);
  if (scored.length < 2) return false;

  const sorted = scored.slice().sort((a, b) => b.speedup - a.speedup);
  const max = Math.max(1.05, sorted[0].speedup) * 1.02;
  const anyLoss = sorted.some((r) => r.speedup < 1);

  const wrap = el('div');
  const bars = el('div', 'bars');
  const pct = (value) => (100 * value / max) + '%';

  sorted.forEach((row) => {
    const line = el('div', 'bar-row');
    line.title = row.label + ' — ' + row.speedup.toFixed(3) + 'x';

    line.appendChild(el('div', 'bar-name', row.label));

    const track = el('div', 'bar-track');
    const fill = el('div', 'bar-fill' + (row.speedup < 1 ? ' is-loss' : ''));
    fill.style.width = pct(row.speedup);
    track.appendChild(fill);
    const parity = el('div', 'bar-parity');
    parity.style.left = pct(1);
    track.appendChild(parity);
    line.appendChild(track);

    line.appendChild(el('div', 'bar-value', row.speedup.toFixed(2) + 'x'));
    bars.appendChild(line);
  });
  wrap.appendChild(bars);

  const scale = el('div', 'bar-scale');
  scale.appendChild(el('span'));
  const axis = el('div', 'bar-axis');
  const ticks = niceTicks(max);
  ticks.forEach((value, index) => {
    const tick = el('span', 'axis-tick', value === 0 ? '0' : value + 'x');
    // Centred ticks would hang off both ends of the axis and get clipped, so
    // the first and last anchor to their edge instead.
    if (index === 0) {
      tick.style.left = '0';
      tick.style.transform = 'none';
    } else if (100 * value / max > 94) {
      tick.style.right = '0';
      tick.style.transform = 'none';
    } else {
      tick.style.left = pct(value);
    }
    axis.appendChild(tick);
  });
  scale.appendChild(axis);
  scale.appendChild(el('span'));
  wrap.appendChild(scale);

  const legend = el('div', 'legend');
  legend.style.marginTop = '14px';
  const item = (cls, text) => {
    const node = el('span', 'legend-item');
    node.appendChild(el('span', 'swatch ' + cls));
    node.appendChild(document.createTextNode(text));
    return node;
  };
  legend.appendChild(item('is-mark', 'faster than the baseline'));
  if (anyLoss) legend.appendChild(item('is-loss', 'slower than the baseline'));
  legend.appendChild(item('is-rule', '1.0x parity'));
  wrap.appendChild(legend);

  container.appendChild(wrap);
  return true;
}

function heroBlock(value, label, tone) {
  const hero = el('div', 'hero');
  hero.appendChild(el('div', 'hero-value' + (tone ? ' is-' + tone : ''), value));
  hero.appendChild(el('div', 'hero-label', label));
  return hero;
}

function tileRow(entries) {
  const tiles = el('div', 'tiles');
  entries.forEach(([value, label, title]) => {
    const tile = el('div', 'tile');
    if (title) tile.title = title;
    tile.appendChild(el('div', 'tile-value', value));
    tile.appendChild(el('div', 'tile-label', label));
    tiles.appendChild(tile);
  });
  return tiles;
}

/* The headline for a set of result rows. Leads with the one number the view is
 * about - a single run's speedup, or a sweep's geometric mean - because that is
 * what someone reads first and the table below carries everything else. */
function renderSummary(container, rows) {
  clear(container);
  const done = rows.filter((r) => r.status === 'done' || r.status === 'failed');
  const scored = rows.filter((r) => typeof r.speedup === 'number');
  const headline = el('div', 'headline');

  if (scored.length === 1) {
    const row = scored[0];
    headline.appendChild(heroBlock(row.speedup.toFixed(2) + 'x',
      'faster than the baseline', row.speedup >= 1 ? 'good' : 'critical'));
    const tiles = [
      [fmt(row.baseline, 3) + ' ms', 'baseline median'],
      [fmt(row.optimized, 3) + ' ms', 'optimized median'],
    ];
    if (row.throughput) tiles.push([Math.round(row.throughput).toLocaleString(), 'token/s']);
    headline.appendChild(tileRow(tiles));
  } else if (scored.length > 1) {
    const mean = geomean(scored.map((r) => r.speedup));
    headline.appendChild(heroBlock(mean.toFixed(2) + 'x',
      'geometric mean over ' + scored.length + ' shapes', mean >= 1 ? 'good' : 'critical'));
    const best = scored.reduce((a, b) => (a.speedup > b.speedup ? a : b));
    const worst = scored.reduce((a, b) => (a.speedup < b.speedup ? a : b));
    headline.appendChild(tileRow([
      [best.speedup.toFixed(2) + 'x', 'best — ' + best.label],
      [worst.speedup.toFixed(2) + 'x', 'worst — ' + worst.label],
    ]));
  } else if (rows.length) {
    const row = rows[0];
    if (row.baselineSkipped && row.optimized) {
      headline.appendChild(heroBlock(fmt(row.optimized, 0) + ' ms',
        row.slices > 1 ? 'per slice, optimized model alone' : 'optimized model alone'));
      const tiles = [];
      if (row.totalMs) tiles.push([fmt(row.totalMs, 0) + ' ms', 'whole batch']);
      if (row.slices > 1) tiles.push([String(row.slices), 'slices']);
      if (row.throughput) tiles.push([Math.round(row.throughput).toLocaleString(), 'token/s']);
      if (tiles.length) headline.appendChild(tileRow(tiles));
    }
  }
  if (headline.childNodes.length) container.appendChild(headline);

  // Accuracy is a verdict, not a magnitude, so it is a status tag rather than a
  // number in the hero row.
  const verdicts = done.filter((r) => r.accuracy !== null);
  if (verdicts.length) {
    const failed = verdicts.filter((r) => r.accuracy === false);
    const strip = el('div', 'btn-row');
    strip.style.marginTop = '16px';
    const tag = el('span', 'tag ' + (failed.length ? 'is-critical' : 'is-good'));
    tag.appendChild(el('span', null, failed.length ? '✖' : '✓'));
    tag.appendChild(document.createTextNode(failed.length
      ? failed.length + ' of ' + verdicts.length + ' failed accuracy'
      : 'accuracy passed on all ' + verdicts.length));
    strip.appendChild(tag);

    const errors = verdicts.map((r) => r.maxAbs).filter((v) => typeof v === 'number');
    if (errors.length) {
      strip.appendChild(el('span', 'status-line',
        'worst max_abs ' + Math.max.apply(null, errors).toExponential(2)));
    }
    container.appendChild(strip);
  }

  if (scored.length > 1) {
    const chart = el('div');
    chart.style.marginTop = '20px';
    if (renderBars(chart, scored)) container.appendChild(chart);
  }
}

/* ------------------------------------------------------------------- log -- */

function appendLog(node, lines, follow) {
  if (!lines || !lines.length) return;
  const atBottom = node.scrollTop + node.clientHeight >= node.scrollHeight - 40;
  lines.forEach((line) => {
    let cls = null;
    if (line.startsWith('[dashboard]')) cls = 'l-meta';
    else if (/\b(FAIL|Error|Traceback|RuntimeError)\b/.test(line)) cls = 'l-bad';
    else if (/\b(PASS|speedup)\b/.test(line)) cls = 'l-good';
    node.appendChild(cls ? el('span', cls, line + '\n') : document.createTextNode(line + '\n'));
  });
  if (follow && atBottom) node.scrollTop = node.scrollHeight;
}

/* ------------------------------------------------------------------ jobs -- */

/* Polling rather than a stream: a 500 ms fetch on a loopback socket is free next
 * to the benchmark it watches, and there is no reconnect logic to get wrong when
 * a run takes twenty minutes. */
function startJob(key, jobId, ui) {
  stopPolling(key);
  state.jobs[key] = jobId;
  clear(ui.log);
  ui.status.textContent = 'queued';
  ui.status.className = 'status-line is-running';
  ui.go.disabled = true;
  ui.stop.disabled = false;

  let since = 0;
  const tick = async () => {
    const { data } = await api('/api/job/' + jobId + '?since=' + since);
    if (!data || data.error) { stopPolling(key); return; }

    if (data.missed_lines) {
      appendLog(ui.log, ['[dashboard] ... ' + data.missed_lines + ' earlier lines '
        + 'dropped from this view; the full log is in dashboard/runs/' + jobId + '.log'],
        ui.follow.checked);
    }
    appendLog(ui.log, data.lines, ui.follow.checked);
    since = data.next_line;

    const rows = data.steps.map(rowFromStep);
    if (rows.length) {
      ui.resultsCard.hidden = false;
      ui.render(rows, data);
    }

    const finished = ['done', 'failed', 'stopped'].includes(data.status);
    const settled = data.steps.filter((s) => ['done', 'failed', 'stopped'].includes(s.status)).length;
    const suffix = data.duration_s !== null ? ' · ' + data.duration_s + 's' : '';
    ui.status.textContent = finished
      ? data.status + suffix
      : data.status + ' · step ' + Math.min(settled + 1, data.steps.length)
        + '/' + data.steps.length + suffix;
    ui.status.className = 'status-line is-' + data.status;

    if (finished) {
      stopPolling(key);
      ui.go.disabled = false;
      ui.stop.disabled = true;
      if (data.error) appendLog(ui.log, ['[dashboard] ' + data.error], true);
      refreshQueue();
    }
  };

  ui.tick = tick;
  tick();
  state.polls[key] = setInterval(tick, 500);
}

function stopPolling(key) {
  if (state.polls[key]) { clearInterval(state.polls[key]); delete state.polls[key]; }
}

/* Stop, with the answer read. Firing the POST and discarding the result made a
 * stop that did not take look exactly like one that did. */
async function requestStop(key, ui) {
  const jobId = state.jobs[key];
  if (!jobId) { ui.status.textContent = 'nothing running to stop'; return; }

  ui.stop.disabled = true;
  const original = ui.stop.textContent;
  ui.stop.textContent = 'stopping…';

  let outcome;
  try {
    outcome = await postJSON('/api/job/' + jobId + '/stop', {});
  } catch (err) {
    outcome = { ok: false, data: { error: String(err) } };
  }
  ui.stop.textContent = original;
  const data = outcome.data || {};

  if (!outcome.ok || data.error) {
    // Leave the button live: whatever went wrong, the job may still be running.
    ui.stop.disabled = false;
    ui.status.textContent = 'stop failed: ' + (data.error || 'HTTP ' + outcome.status);
    ui.status.className = 'status-line is-failed';
    appendLog(ui.log, ['[dashboard] stop failed: ' + (data.error || outcome.status)], true);
    return;
  }
  if (data.stopped === false) {
    ui.status.textContent = 'already finished';
    appendLog(ui.log, ['[dashboard] stop had nothing to do; the job had already finished'], true);
    return;
  }
  appendLog(ui.log, ['[dashboard] stop requested; killing the process tree'], true);
  if (state.polls[key]) {
    clearInterval(state.polls[key]);
    state.polls[key] = setInterval(ui.tick, 500);
  }
  if (ui.tick) ui.tick();
}

async function submit(key, payload, ui) {
  const { ok, data } = await postJSON('/api/run', payload);
  if (!ok || !data || data.error) {
    renderIssues(ui.issues, (data && data.issues) || [
      { level: 'error', message: (data && data.error) || 'the server rejected this run' },
    ]);
    return;
  }
  if (data.issues && data.issues.length) renderIssues(ui.issues, data.issues);
  startJob(key, data.id, ui);
  refreshQueue();
}

/* -------------------------------------------------------------- run view -- */

const QUICK = { warmup: 5, repeats: 20, benchmark_rounds: 1, accuracy_trials: 2 };

function wireSpeedButtons() {
  document.querySelectorAll('[data-speed]').forEach((button) => {
    button.addEventListener('click', () => {
      const quick = button.dataset.speed === 'quick';
      Object.keys(QUICK).forEach((dest) => {
        const container = $('run-timing').querySelector('[data-dest="' + dest + '"]')
          ? $('run-timing') : $('run-accuracy');
        setFieldValue(container, dest, quick ? QUICK[dest] : '');
      });
    });
  });
}

function setupRun() {
  const form = state.forms.run;

  const updateEnvBadge = () => {
    const count = Object.keys(form.env || {}).length;
    const badge = $('run-env-count');
    badge.textContent = count + ' set';
    badge.hidden = !count;
  };
  const preflight = makePreflight(() => form, $('run-command'), $('run-issues'),
    $('run-derived'), updateEnvBadge);
  const changed = () => preflight();

  ['shape', 'optimization', 'timing', 'accuracy', 'data', 'torch', 'other']
    .forEach((group) => renderGroup($('run-' + group), group, form, changed));
  renderEnv($('run-env'), form, changed);

  fillPresetSelect($('run-preset'), (preset) => applyPreset($('run-shape'), preset));
  wireSpeedButtons();

  $('run-copy-cmd').addEventListener('click', () => {
    navigator.clipboard.writeText($('run-command').textContent);
  });

  const ui = {
    log: $('run-log'), status: $('run-job-status'), go: $('run-go'), stop: $('run-stop'),
    follow: $('run-follow'), issues: $('run-issues'), resultsCard: $('run-results-card'),
    render: (rows) => { renderSummary($('run-summary'), rows); renderTable($('run-results'), rows); },
  };

  $('run-go').addEventListener('click', () => submit('run', { mode: 'single', form }, ui));
  $('run-stop').addEventListener('click', () => requestStop('run', ui));
  preflight();
}

/* ---------------------------------------------------------- compare view -- */

function mergedCompareForm(side) {
  return Object.assign({}, state.forms.cmpShape, side, { env: side.env || {} });
}

function setupCompare() {
  const a = state.forms.cmpA;
  const b = state.forms.cmpB;

  // Both sides are checked live and their issues merged, prefixed the way the
  // server prefixes them on submit - otherwise a setting that only breaks B
  // stays invisible until the run is rejected.
  const sideIssues = { A: [], B: [] };
  const showIssues = () => renderIssues($('cmp-issues'),
    sideIssues.A.map((i) => Object.assign({}, i, { message: 'A: ' + i.message }))
      .concat(sideIssues.B.map((i) => Object.assign({}, i, { message: 'B: ' + i.message }))));

  const preA = makePreflight(() => mergedCompareForm(a), $('cmp-a-command'), null,
    $('cmp-derived'), (data) => { sideIssues.A = data.issues || []; showIssues(); });
  const preB = makePreflight(() => mergedCompareForm(b), $('cmp-b-command'), null,
    null, (data) => { sideIssues.B = data.issues || []; showIssues(); });
  const both = () => { preA(); preB(); };

  renderGroup($('cmp-shape'), 'shape', state.forms.cmpShape, both);
  renderGroup($('cmp-timing'), 'timing', state.forms.cmpShape, both);
  renderGroup($('cmp-a-optimization'), 'optimization', a, both);
  renderGroup($('cmp-b-optimization'), 'optimization', b, both);
  renderEnv($('cmp-a-env'), a, both);
  renderEnv($('cmp-b-env'), b, both);

  fillPresetSelect($('cmp-preset'), (preset) => applyPreset($('cmp-shape'), preset));

  $('cmp-copy-ab').addEventListener('click', () => {
    state.spec.fields.filter((spec) => spec.group === 'optimization').forEach((spec) =>
      setFieldValue($('cmp-b-optimization'), spec.dest,
        a[spec.dest] === undefined ? '' : a[spec.dest]));
    state.spec.env_knobs.forEach((knob) => {
      const value = (a.env || {})[knob.name];
      setFieldValue($('cmp-b-env'), knob.name, value === undefined ? knob.default : value);
    });
  });

  const ui = {
    log: $('cmp-log'), status: $('cmp-job-status'), go: $('cmp-go'), stop: $('cmp-stop'),
    follow: $('cmp-follow'), issues: $('cmp-issues'), resultsCard: $('cmp-results-card'),
    render: renderCompareResults,
  };

  $('cmp-go').addEventListener('click', () => submit('compare', {
    mode: 'compare',
    form_a: mergedCompareForm(a),
    form_b: mergedCompareForm(b),
    control: $('cmp-control').checked,
  }, ui));
  $('cmp-stop').addEventListener('click', () => requestStop('compare', ui));
  both();
}

/* B against A, with the control's own deviation from 1.000x as the bar the
 * difference has to clear before it is called a result. */
function renderCompareResults(rows) {
  const find = (role) => rows.find((row) => row.role === role);
  const base = find('A');

  renderTable($('cmp-results'), rows, [{
    key: 'vsA', title: 'vs A',
    render: (cell, row) => {
      if (!base || row.role === 'A' || !base.optimized || !row.optimized) {
        cell.textContent = '–'; cell.classList.add('is-muted'); return;
      }
      const ratio = base.optimized / row.optimized;
      cell.textContent = ratio.toFixed(3) + 'x';
      cell.classList.add('is-strong', ratio >= 1 ? 'is-good' : 'is-critical');
    },
  }]);

  const summary = $('cmp-summary');
  clear(summary);
  const target = find('B');
  const control = find('control');
  if (!base || !target || !base.optimized || !target.optimized) return;

  const ratio = base.optimized / target.optimized;
  const gapPct = Math.abs(ratio - 1) * 100;

  const headline = el('div', 'headline');
  headline.appendChild(heroBlock(ratio.toFixed(3) + 'x', 'B relative to A'));
  const tiles = [
    [fmt(base.optimized, 3) + ' ms', 'A median'],
    [fmt(target.optimized, 3) + ' ms', 'B median'],
  ];
  if (control && control.optimized) tiles.push([fmt(control.optimized, 3) + ' ms', 'control median']);
  headline.appendChild(tileRow(tiles));
  summary.appendChild(headline);

  const verdict = el('div', 'verdict');
  verdict.style.marginTop = '18px';

  if (control && control.optimized) {
    const controlRatio = base.optimized / control.optimized;
    const noisePct = Math.abs(controlRatio - 1) * 100;
    const line = el('div');
    line.innerHTML = 'B is <b>' + ratio.toFixed(3) + 'x</b> config A, a gap of '
      + gapPct.toFixed(1) + '%. The control &mdash; A run twice, true value 1.000x '
      + '&mdash; came back at <b>' + controlRatio.toFixed(3) + 'x</b>, so the noise '
      + 'floor on this machine right now is ' + noisePct.toFixed(1) + '%.';
    verdict.appendChild(line);

    const call = el('div', 'v-call');
    let tone, glyph, text;
    if (gapPct <= noisePct) {
      tone = 'is-warning'; glyph = '⚠';
      text = 'Inside the noise. This run does not separate A from B — raise '
        + '--benchmark-rounds or --repeats, or accept that they measure the same.';
    } else if (gapPct < noisePct * 2) {
      tone = 'is-warning'; glyph = '⚠';
      text = 'Clears the control, but by less than 2x. Worth repeating before trusting.';
    } else {
      tone = ratio > 1 ? 'is-critical' : 'is-good';
      glyph = ratio > 1 ? '✖' : '✓';
      text = ratio > 1
        ? 'B is the slower of the two, by comfortably more than the noise floor.'
        : 'B is the faster of the two, by comfortably more than the noise floor.';
    }
    const tag = el('span', 'tag ' + tone);
    tag.appendChild(el('span', null, glyph));
    tag.appendChild(document.createTextNode(gapPct <= noisePct ? 'not a result' : 'real'));
    call.appendChild(tag);
    call.appendChild(el('span', null, text));
    verdict.appendChild(call);
  } else {
    const line = el('div');
    line.innerHTML = 'B is <b>' + ratio.toFixed(3) + 'x</b> config A, a gap of '
      + gapPct.toFixed(1) + '%. No control was run, so there is nothing to say '
      + "whether that gap is real: this project's own tuning notes put the noise "
      + 'floor near 4%.';
    verdict.appendChild(line);
  }
  summary.appendChild(verdict);
}

/* ------------------------------------------------------------ sweep view -- */

function setupSweep() {
  const form = state.forms.sweep;
  const preflight = makePreflight(() => form, null, $('sweep-issues'), null);

  renderGroup($('sweep-optimization'), 'optimization', form, preflight);
  renderGroup($('sweep-timing'), 'timing', form, preflight);
  renderEnv($('sweep-env'), form, preflight);

  const list = $('sweep-shapes');
  $('sweep-presets-path').textContent = state.spec.presets_path.split(/[\\/]/).slice(-2).join('/');

  // Named so it can be re-run: saving in the Presets view changes which shapes
  // exist, and a list still offering the old ones would queue shapes that are no
  // longer in the file.
  const buildList = () => {
    clear(list);
    state.spec.presets.forEach((preset, index) => {
      const row = el('label', 'pick' + (preset.blocked ? ' is-blocked' : ''));

      const box = el('input');
      box.type = 'checkbox';
      box.checked = !preset.blocked;
      box.disabled = !!preset.blocked;
      box.dataset.index = String(index);
      row.appendChild(box);

      const names = el('div');
      names.appendChild(el('div', 'pick-name', preset.name));
      names.appendChild(el('div', 'pick-dims', [
        preset.batch_size !== undefined ? 'b' + preset.batch_size : null,
        preset.seq_len !== undefined ? 's' + preset.seq_len : null,
        preset.d_model !== undefined ? 'd' + preset.d_model : null,
        preset.heads !== undefined ? 'h' + preset.heads : null,
        preset.layers !== undefined ? 'L' + preset.layers : null,
      ].filter(Boolean).join(' ')));
      row.appendChild(names);

      row.appendChild(el('div', 'pick-meta',
        (preset.head_dim ? 'hd ' + preset.head_dim : '')));
      row.appendChild(el('div', 'pick-meta', preset.memory || ''));

      // What the harness will do with the shape, not just whether it runs: both
      // of these change what the row's numbers mean.
      const tags = el('div', 'pick-tags');
      if (preset.slices > 1) {
        const tag = el('span', 'tag is-info', preset.slices + ' slices');
        tag.title = 'The full input does not fit, so the harness streams the batch. '
          + 'Same computation; the median is per slice.';
        tags.appendChild(tag);
      }
      if (preset.baseline_runs === false) {
        const tag = el('span', 'tag is-warning', 'no baseline');
        tag.title = 'The baseline cannot hold its score matrix at this shape, so it '
          + 'is skipped. You get latencies but no speedup.';
        tags.appendChild(tag);
      }
      if (preset.blocked) {
        const tag = el('span', 'tag is-critical', 'cannot run');
        tag.title = preset.blocked_reason;
        tags.appendChild(tag);
      }
      row.appendChild(tags);

      row.title = [preset.note, preset.blocked_reason]
        .concat(preset.notes || []).filter(Boolean).join('\n\n');
      list.appendChild(row);
    });
  };

  const setAll = (checked) => list.querySelectorAll('input')
    .forEach((box) => { if (!box.disabled) box.checked = checked; });

  const updateCount = () => {
    const chosen = list.querySelectorAll('input:checked').length;
    const blocked = state.spec.presets.filter((p) => p.blocked).length;
    $('sweep-count').textContent = chosen === 0
      ? 'nothing selected'
      : chosen + ' shape' + (chosen === 1 ? '' : 's') + ' selected'
        + (blocked ? ' · ' + blocked + ' cannot run' : '');
  };
  list.addEventListener('change', updateCount);

  buildList();
  rebuildShapeList = () => { buildList(); updateCount(); };

  $('sweep-all').addEventListener('click', () => { setAll(true); updateCount(); });
  $('sweep-none').addEventListener('click', () => { setAll(false); updateCount(); });
  updateCount();

  const ui = {
    log: $('sweep-log'), status: $('sweep-job-status'), go: $('sweep-go'),
    stop: $('sweep-stop'), follow: $('sweep-follow'), issues: $('sweep-issues'),
    resultsCard: $('sweep-results-card'),
    render: (rows) => { renderSummary($('sweep-summary'), rows); renderTable($('sweep-results'), rows); },
  };

  $('sweep-go').addEventListener('click', () => {
    const shapes = Array.from(list.querySelectorAll('input:checked'))
      .map((box) => state.spec.presets[Number(box.dataset.index)]);
    if (!shapes.length) {
      renderIssues($('sweep-issues'), [{ level: 'error', message: 'no shapes ticked' }]);
      return;
    }
    submit('sweep', { mode: 'sweep', form, shapes }, ui);
  });
  $('sweep-stop').addEventListener('click', () => requestStop('sweep', ui));
  preflight();
}

/* ---------------------------------------------------------- scripts view -- */

function setupScripts() {
  const list = $('script-list');
  clear(list);

  state.spec.scripts.forEach((script) => {
    const row = el('div', 'script');
    row.setAttribute('role', 'button');
    row.setAttribute('aria-selected', 'false');

    const name = el('div', 'script-name');
    name.appendChild(document.createTextNode(script.name));
    if (script.long_running) name.appendChild(el('span', 'tag is-warning', 'slow'));
    if (script.has_args) name.appendChild(el('span', 'tag is-quiet', script.fields.length + ' args'));
    row.appendChild(name);
    row.appendChild(el('div', 'script-desc', script.summary || 'no description'));
    row.addEventListener('click', () => selectScript(script, row));
    list.appendChild(row);
  });

  $('script-extra').addEventListener('input', updateScriptCommand);

  const ui = {
    log: $('script-log'), status: $('script-job-status'), go: $('script-go'),
    stop: $('script-stop'), follow: $('script-follow'),
    issues: el('div'), resultsCard: el('div'), render: () => {},
  };

  $('script-go').addEventListener('click', () => {
    if (!state.selectedScript) return;
    submit('script', {
      mode: 'script',
      script: state.selectedScript.name,
      form: state.forms.script,
      extra: $('script-extra').value,
    }, ui);
  });
  $('script-stop').addEventListener('click', () => requestStop('script', ui));
}

function selectScript(script, row) {
  document.querySelectorAll('.script').forEach((node) => node.setAttribute('aria-selected', 'false'));
  row.setAttribute('aria-selected', 'true');

  state.selectedScript = script;
  state.forms.script = {};

  $('script-title').textContent = script.name;
  $('script-summary').textContent = script.summary
    + (script.long_running
      ? '  — this one is slow; tune_block_shapes builds a separate CUDA extension per candidate.'
      : '');
  $('script-go').disabled = false;

  const container = $('script-fields');
  clear(container);
  script.fields.forEach((spec) =>
    container.appendChild(makeField(spec, state.forms.script, updateScriptCommand)));
  if (!script.fields.length) {
    container.appendChild(el('p', 'prose',
      'This script takes no command-line arguments. Use the box below to pass '
      + 'something anyway.'));
  }
  updateScriptCommand();
}

function updateScriptCommand() {
  if (!state.selectedScript) return;
  const parts = ['python', '-u', 'scripts/' + state.selectedScript.name];
  (state.selectedScript.fields || []).forEach((spec) => {
    const value = state.forms.script[spec.dest];
    if (value === undefined) return;
    if (spec.kind === 'flag') { if (value) parts.push(spec.flag); return; }
    parts.push(spec.flag, String(value));
  });
  const extra = $('script-extra').value.trim();
  if (extra) parts.push(extra);
  $('script-command').innerHTML = highlightCommand(parts.join(' '));
}

/* ---------------------------------------------------------- presets view -- */

const PRESET_COLUMNS = [
  { key: 'name', label: 'name', type: 'text', cls: 'col-name' },
  { key: 'batch_size', label: 'batch', type: 'number', cls: 'col-num' },
  { key: 'seq_len', label: 'seq_len', type: 'number', cls: 'col-num' },
  { key: 'd_model', label: 'd_model', type: 'number', cls: 'col-num' },
  { key: 'heads', label: 'heads', type: 'number', cls: 'col-num' },
  { key: 'ffn_dim', label: 'ffn_dim', type: 'number', cls: 'col-num' },
  { key: 'layers', label: 'layers', type: 'number', cls: 'col-num' },
  { key: 'causal', label: 'causal', type: 'checkbox', cls: 'col-num' },
  { key: 'note', label: 'note', type: 'text', cls: 'col-note' },
];

const presetState = { rows: [], dirty: false };

function presetRowFrom(preset) {
  const row = {};
  PRESET_COLUMNS.forEach((column) => {
    row[column.key] = preset[column.key] === undefined
      ? (column.type === 'checkbox' ? false : '') : preset[column.key];
  });
  return row;
}

function markPresetsDirty(dirty) {
  presetState.dirty = dirty;
  const node = $('presets-status');
  node.textContent = dirty ? 'unsaved changes' : '';
  node.className = 'status-line' + (dirty ? ' is-stopped' : '');
}

/* head_dim per row while typing. It is the derived fact that decides whether a
 * shape can use the custom kernels at all, and save time is later than needed. */
function presetDerived(row) {
  const d = Number(row.d_model), h = Number(row.heads);
  if (!d || !h) return ['', false];
  if (d % h) return ['d_model % heads ≠ 0', true];
  return ['hd ' + (d / h), false];
}

function renderPresetTable(problems) {
  const table = $('presets-table');
  clear(table);
  const byField = {};
  (problems || []).forEach((p) => { byField[p.row + '|' + p.field] = p.message; });

  const head = table.createTHead().insertRow();
  PRESET_COLUMNS.forEach((column) => head.appendChild(el('th', null, column.label)));
  head.appendChild(el('th', null, 'derived'));
  head.appendChild(el('th', null, ''));

  const body = table.createTBody();
  presetState.rows.forEach((row, index) => {
    const tr = body.insertRow();

    PRESET_COLUMNS.forEach((column) => {
      const cell = tr.insertCell();
      cell.className = column.cls || '';
      const input = el('input');
      input.type = column.type;
      if (column.type === 'checkbox') input.checked = !!row[column.key];
      else input.value = row[column.key] === undefined ? '' : row[column.key];
      if (column.type === 'number') input.min = '1';

      const problem = byField[index + '|' + column.key];
      if (problem) { input.classList.add('is-bad'); input.title = problem; }

      const sync = () => {
        if (column.type === 'checkbox') row[column.key] = input.checked;
        else if (column.type === 'number') row[column.key] = input.value === '' ? '' : Number(input.value);
        else row[column.key] = input.value;
        input.classList.remove('is-bad');
        const cellNode = tr.querySelector('.cell-derived');
        if (cellNode) {
          const [text, bad] = presetDerived(row);
          cellNode.textContent = text;
          cellNode.classList.toggle('is-bad', bad);
        }
        markPresetsDirty(true);
      };
      input.addEventListener('input', sync);
      input.addEventListener('change', sync);
      cell.appendChild(input);
    });

    const derivedCell = tr.insertCell();
    const [text, bad] = presetDerived(row);
    derivedCell.appendChild(el('span', 'cell-derived' + (bad ? ' is-bad' : ''), text));

    const actions = tr.insertCell();
    const drop = el('button', 'icon-btn', '×');
    drop.title = 'remove this shape';
    drop.addEventListener('click', () => {
      presetState.rows.splice(index, 1);
      markPresetsDirty(true);
      renderPresetTable();
    });
    actions.appendChild(drop);
  });
}

function loadPresetRows() {
  presetState.rows = state.spec.presets.map(presetRowFrom);
  markPresetsDirty(false);
  renderPresetTable();
  renderIssues($('presets-issues'), []);
}

/* Every control fed by presets, rebuilt after a save; otherwise they keep
 * offering shapes that no longer exist. */
function refreshPresetConsumers() {
  ['run-preset', 'cmp-preset'].forEach((id) => {
    const select = $(id);
    if (!select) return;
    while (select.options.length > 1) select.remove(1);
    state.spec.presets.forEach((preset, index) => {
      const option = new Option(preset.name, String(index));
      if (preset.note) option.title = preset.note;
      select.appendChild(option);
    });
  });
  if (rebuildShapeList) rebuildShapeList();
}

function setupPresets() {
  $('presets-path').textContent = state.spec.presets_path.split(/[\\/]/).slice(-2).join('/');
  loadPresetRows();

  $('presets-add').addEventListener('click', () => {
    // Seeded from the last row rather than blank: shapes in this set differ from
    // each other by one axis, so copy-and-change-one-number is what adding one
    // usually means.
    const last = presetState.rows[presetState.rows.length - 1];
    presetState.rows.push(last
      ? Object.assign({}, last, { name: '', note: '' })
      : { name: '', batch_size: 8, seq_len: 128, d_model: 512, heads: 8,
          ffn_dim: 512, layers: 6, causal: true, note: '' });
    markPresetsDirty(true);
    renderPresetTable();
    const inputs = $('presets-table').querySelectorAll('tbody tr:last-child input');
    if (inputs.length) inputs[0].focus();
  });

  $('presets-revert').addEventListener('click', () => {
    if (presetState.dirty && !confirm('Discard the unsaved changes and reload presets.json?')) return;
    loadPresetRows();
  });

  $('presets-save').addEventListener('click', async () => {
    const button = $('presets-save');
    button.disabled = true;
    $('presets-status').textContent = 'saving…';

    const payload = presetState.rows.map((row) => {
      const out = {};
      PRESET_COLUMNS.forEach((column) => {
        if (column.type === 'checkbox') out[column.key] = !!row[column.key];
        else if (column.type === 'number') out[column.key] = row[column.key] === '' ? null : Number(row[column.key]);
        else out[column.key] = row[column.key];
      });
      return out;
    });

    const { ok, data } = await postJSON('/api/presets', { presets: payload });
    button.disabled = false;

    if (!ok || !data || data.error) {
      const problems = (data && data.problems) || [];
      renderPresetTable(problems);
      renderIssues($('presets-issues'), problems.length
        ? problems.map((p) => ({ level: 'error',
            message: 'row ' + (p.row + 1) + (p.field ? ' · ' + p.field : '') + ': ' + p.message }))
        : [{ level: 'error', message: (data && data.error) || 'save failed' }]);
      const status = $('presets-status');
      status.textContent = 'not saved';
      status.className = 'status-line is-failed';
      return;
    }

    state.spec.presets = data.presets;
    loadPresetRows();
    refreshPresetConsumers();
    const status = $('presets-status');
    status.textContent = 'saved ' + data.saved + ' shapes';
    status.className = 'status-line is-done';
  });
}

/* ---------------------------------------------------------- history view -- */

async function refreshHistory() {
  const { data } = await api('/api/history?limit=60');
  const table = $('history-table');
  clear(table);
  const entries = (data && data.entries) || [];

  const head = table.createTHead().insertRow();
  ['when', 'job', 'mode', 'status', 'steps', 'best speedup', 'took']
    .forEach((title, index) => head.appendChild(el('th', index < 4 ? 'left' : null, title)));

  const body = table.createTBody();
  if (!entries.length) {
    const cell = body.insertRow().insertCell();
    cell.colSpan = 7;
    const empty = el('div', 'empty');
    empty.appendChild(el('div', 'empty-glyph', '○'));
    empty.appendChild(el('div', 'empty-title', 'No finished jobs yet'));
    empty.appendChild(el('div', 'empty-hint',
      'Runs are recorded here once they finish, with their full logs kept beside '
      + 'the index in dashboard/runs/.'));
    cell.appendChild(empty);
    return;
  }

  entries.forEach((entry) => {
    const tr = body.insertRow();
    const speedups = (entry.steps || [])
      .map((step) => step.result && step.result.speedup)
      .filter((value) => typeof value === 'number');
    const best = speedups.length ? Math.max.apply(null, speedups) : null;

    [
      new Date(entry.created_at * 1000).toLocaleString(),
      entry.id,
      entry.mode,
      entry.status,
      String((entry.steps || []).length),
      best === null ? '–' : best.toFixed(3) + 'x',
      entry.duration_s === null ? '–' : entry.duration_s + 's',
    ].forEach((text, index) => {
      const cell = tr.insertCell();
      if (index < 4) cell.className = 'left';
      cell.textContent = text;
      if (index === 3) {
        cell.classList.add(entry.status === 'done' ? 'is-good'
          : (entry.status === 'failed' ? 'is-critical' : 'is-muted'));
      }
      if (index === 5 && best !== null) {
        cell.classList.add(best >= 1 ? 'is-good' : 'is-critical');
      }
    });
    tr.title = entry.title || '';
  });
}

/* ---------------------------------------------------------------- chrome -- */

function fillPresetSelect(select, onPick) {
  state.spec.presets.forEach((preset, index) => {
    const option = new Option(preset.name, String(index));
    if (preset.note) option.title = preset.note;
    select.appendChild(option);
  });
  select.addEventListener('change', () => {
    if (select.value === '') return;
    onPick(state.spec.presets[Number(select.value)]);
  });
}

function applyPreset(container, preset) {
  state.spec.shape_keys.forEach((key) => {
    if (preset[key] === undefined) return;
    setFieldValue(container, key, preset[key]);
  });
}

async function refreshGPU() {
  const { data } = await api('/api/gpu');
  if (!data) return;
  if (!data.available) {
    $('gpu-name').textContent = 'nvidia-smi unavailable';
    $('gpu-mem').textContent = 'GPU telemetry off';
    return;
  }
  const pct = Math.round(100 * data.memory_used_mib / data.memory_total_mib);
  $('gpu-name').textContent = data.name;
  $('gpu-util').textContent = data.utilization + '%';
  $('gpu-mem').textContent = data.memory_used_mib.toLocaleString() + ' / '
    + data.memory_total_mib.toLocaleString() + ' MiB';
  $('gpu-temp').textContent = data.temperature + '°C';
  const meter = $('gpu-meter');
  meter.style.width = pct + '%';
  meter.className = 'track-fill' + (pct > 90 ? ' is-critical' : (pct > 70 ? ' is-warning' : ''));
  $('gpu-chip').title = 'GPU memory in use across the whole machine, not just this '
    + 'dashboard. The server itself holds no CUDA context.';
}

async function refreshQueue() {
  const { data } = await api('/api/queue');
  if (!data) return;
  const pill = $('queue-pill');
  if (data.running) {
    $('queue-text').textContent = data.queued
      ? 'running ' + data.running + ' · ' + data.queued + ' queued'
      : 'running ' + data.running;
    pill.className = 'queue-pill is-busy';
  } else {
    $('queue-text').textContent = data.queued ? data.queued + ' queued' : 'idle';
    pill.className = 'queue-pill';
  }
}

function showExtension(info) {
  const dot = $('ext-dot');
  const text = $('ext-text');
  const chip = $('ext-chip');
  if (!info || !info.probed) {
    dot.className = 'dot';
    text.textContent = 'extension: not probed';
    chip.title = 'Probing builds and loads the CUDA extension in a throwaway '
      + 'process. On a cold tree that takes about 70 seconds.';
    return;
  }
  if (!info.loaded) {
    dot.className = 'dot is-critical';
    text.textContent = 'extension: not loaded';
  } else if (info.tile) {
    dot.className = 'dot is-good';
    text.textContent = 'loaded, with cuTile';
  } else {
    dot.className = 'dot is-warning';
    text.textContent = 'loaded, no cuTile';
  }
  chip.title = info.error || 'The tile-* attention modes need a cuTile-capable '
    + 'build (CUDA 13.3+).';
}

function showView(name) {
  document.querySelectorAll('.nav-item').forEach((tab) =>
    tab.setAttribute('aria-selected', String(tab.dataset.tab === name)));
  document.querySelectorAll('.view').forEach((view) =>
    view.classList.toggle('is-active', view.id === 'panel-' + name));
  const [title, sub] = VIEW_TITLES[name] || [name, ''];
  $('view-title').textContent = title;
  $('view-sub').textContent = sub;
  if (name === 'history') refreshHistory();
  // Reloading here picks up an edit made in a text editor while the page was
  // open, without clobbering work in progress.
  if (name === 'presets' && !presetState.dirty) loadPresetRows();
}

function setupChrome() {
  document.querySelectorAll('.nav-item').forEach((tab) => {
    tab.addEventListener('click', () => showView(tab.dataset.tab));
  });

  document.querySelectorAll('[data-theme-choice]').forEach((button) => {
    button.addEventListener('click', () => applyTheme(button.dataset.themeChoice));
  });

  document.querySelectorAll('[data-csv]').forEach((button) => {
    button.addEventListener('click', () => {
      navigator.clipboard.writeText(tableToCSV($(button.dataset.csv)));
      const original = button.textContent;
      button.textContent = 'copied';
      setTimeout(() => { button.textContent = original; }, 1200);
    });
  });

  $('history-refresh').addEventListener('click', refreshHistory);

  $('ext-probe').addEventListener('click', async () => {
    const button = $('ext-probe');
    button.disabled = true;
    $('ext-text').textContent = 'probing…';
    const { data } = await postJSON('/api/probe');
    if (data && data.error && !data.probed) {
      $('ext-text').textContent = 'probe unavailable';
      $('ext-chip').title = data.error;
    } else {
      state.spec.extension = data;
      showExtension(data);
    }
    button.disabled = false;
  });

  // Alt+1..6 moves between views without leaving the keyboard.
  const order = ['run', 'compare', 'sweep', 'scripts', 'presets', 'history'];
  window.addEventListener('keydown', (event) => {
    if (!event.altKey || event.ctrlKey || event.metaKey) return;
    const index = Number(event.key) - 1;
    if (index >= 0 && index < order.length) { showView(order[index]); event.preventDefault(); }
  });
}

/* ------------------------------------------------------------------ boot -- */

async function boot() {
  const { data } = await api('/api/spec');
  if (!data) {
    document.body.innerHTML = '<p style="padding:40px;font:14px system-ui">'
      + 'Could not reach the dashboard server. Is <code>python -m dashboard</code> '
      + 'still running?</p>';
    return;
  }
  state.spec = data;
  $('repo-path').textContent = data.repo;
  showExtension(data.extension);

  setupChrome();
  setupRun();
  setupCompare();
  setupSweep();
  setupScripts();
  setupPresets();

  const warnings = [];
  if (data.presets_error) warnings.push({ level: 'warning', message: data.presets_error });
  if (data.flag_source === 'fallback') {
    warnings.push({ level: 'warning',
      message: 'The harness’s flags could not be read from its source, so a '
        + 'built-in list is in use. It may be out of date; check that parse_args() '
        + 'still lives in torch_transformer_benchmark.py.' });
  }
  if (warnings.length) renderIssues($('run-issues'), warnings);

  refreshGPU();
  refreshQueue();
  setInterval(refreshGPU, 2000);
  setInterval(refreshQueue, 3000);
}

boot();

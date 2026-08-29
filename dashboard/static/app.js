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
  run: ['Run', 'One configuration, on one shape or many'],
  compare: ['Compare', 'Two configurations, timed back to back against a control'],
  profile: ['Profile', 'One traced run, and where its GPU time actually goes'],
  scripts: ['Scripts', "The repository's own A/B and verification scripts"],
  presets: ['Presets', 'The saved shapes, written to presets.json'],
  history: ['History', 'Finished jobs and where their logs are'],
};

const state = {
  spec: null,
  forms: { run: {}, cmpShape: {}, cmpA: {}, cmpB: {}, script: {}, profile: {} },
  jobs: { run: null, compare: null, script: null, profile: null, ncu: null },
  polls: {},
  // Each view's Run/Stop pair, so the queue poll can put them back if a job's
  // own polling ever stops reporting. See reconcileControls().
  uis: {},
  selectedScript: null,
};

// Every shape list that exists, so saving a preset rebuilds all of them. A list
// still offering a shape that has been deleted from the file would queue it.
const shapeLists = [];

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
      // No "auto: " prefix: the select directly above already says which value
      // this describes, and on one clamped line those six characters are a
      // third of what there is room for.
      note.textContent = text || '';
      // Clamped in CSS, so the whole of it lives here instead -- these notes
      // carry measured numbers that are worth keeping reachable.
      if (text) note.title = active + ': ' + text; else note.removeAttribute('title');
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
    preset: (step.meta && step.meta.preset) || '',
    shape: (step.meta && step.meta.shape) || '',
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

/* `type` drives both sorting and what the filter offers: "num" columns get the
 * numeric predicate, everything else sorts as text. */
const COLUMNS = [
  { key: 'label', title: 'run', left: true, type: 'text' },
  { key: 'status', title: 'state', type: 'text' },
  { key: 'accuracy', title: 'accuracy', type: 'text' },
  { key: 'maxAbs', title: 'max_abs', type: 'num' },
  { key: 'maxRel', title: 'max_rel', type: 'num' },
  { key: 'failed', title: 'failed', type: 'text' },
  { key: 'baseline', title: 'baseline ms', type: 'num' },
  { key: 'optimized', title: 'optimized ms', type: 'num' },
  { key: 'p90', title: 'p90 ms', type: 'num' },
  { key: 'min', title: 'min ms', type: 'num' },
  { key: 'speedup', title: 'speedup', type: 'num' },
  { key: 'throughput', title: 'token/s', type: 'num' },
  { key: 'secs', title: 'took', type: 'num' },
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

/* Per-table view state: the rows as they arrived, plus how this table is being
 * looked at. Kept so a filter can be re-applied without another fetch, and so a
 * running job's poll does not reset the sort under the reader's cursor. */
const tableViews = {};

const BLANK_VIEW = () => ({
  sort: { key: null, dir: 'desc' },
  text: '', accuracy: '', status: '', numKey: 'speedup', numOp: 'ge', numValue: '',
});

function viewFor(id) {
  if (!tableViews[id]) {
    tableViews[id] = Object.assign({ id, rows: [], columns: [] }, BLANK_VIEW());
  }
  return tableViews[id];
}

function viewIsFiltering(view) {
  return !!(view.text || view.accuracy || view.status
    || (view.numValue !== '' && view.numValue !== null));
}

function activeFilterCount(view) {
  return [view.text, view.accuracy, view.status,
    (view.numValue === '' || view.numValue === null) ? '' : 'n'].filter(Boolean).length;
}

/* The value a column sorts and filters on, which is not always what the cell
 * prints: "optimized ms" shows a suffix on a streamed row, and the compare
 * view's "vs A" is computed rather than stored. */
function cellValue(column, row) {
  if (column.value) return column.value(row);
  if (column.key === 'accuracy') {
    return row.accuracy === null || row.accuracy === undefined
      ? '' : (row.accuracy ? 'PASS' : 'FAIL');
  }
  return row[column.key];
}

function applyView(view) {
  const byKey = {};
  view.columns.forEach((column) => { byKey[column.key] = column; });
  let rows = view.rows.slice();

  if (view.text) {
    const needle = view.text.toLowerCase();
    rows = rows.filter((row) => String(row.label || '').toLowerCase().includes(needle));
  }
  if (view.accuracy) {
    const want = view.accuracy === 'pass';
    rows = rows.filter((row) => row.accuracy === want);
  }
  if (view.status) {
    rows = rows.filter((row) => row.status === view.status);
  }
  if (view.numValue !== '' && view.numValue !== null && byKey[view.numKey]) {
    const threshold = Number(view.numValue);
    if (!Number.isNaN(threshold)) {
      rows = rows.filter((row) => {
        const value = cellValue(byKey[view.numKey], row);
        // A row with no value for the column cannot satisfy a numeric test, and
        // silently keeping it would misreport the filter.
        if (typeof value !== 'number' || Number.isNaN(value)) return false;
        return view.numOp === 'ge' ? value >= threshold : value <= threshold;
      });
    }
  }

  const column = byKey[view.sort.key];
  if (column) {
    const sign = view.sort.dir === 'asc' ? 1 : -1;
    rows.sort((a, b) => {
      const x = cellValue(column, a);
      const y = cellValue(column, b);
      const xEmpty = x === null || x === undefined || x === '';
      const yEmpty = y === null || y === undefined || y === '';
      // Rows with nothing in the sorted column sink to the bottom either way -
      // a run with no speedup is not "the smallest speedup".
      if (xEmpty && yEmpty) return 0;
      if (xEmpty) return 1;
      if (yEmpty) return -1;
      if (typeof x === 'number' && typeof y === 'number') return (x - y) * sign;
      return String(x).localeCompare(String(y), undefined, { numeric: true }) * sign;
    });
  }
  return rows;
}

/* `baseColumns` replaces the standard set rather than extending it, for a table
 * whose rows are not one-run-per-row -- the compare pivot, where a row is a shape
 * and the run columns appear once per config. Sorting and filtering read
 * view.columns, so they follow whatever is passed. */
function renderTable(table, rows, extraColumns, baseColumns) {
  const view = viewFor(table.id);
  view.rows = rows;
  view.columns = (baseColumns || COLUMNS).concat(extraColumns || []);
  paintTable(view);
}

function paintTable(view) {
  const table = $(view.id);
  if (!table) return;
  const shown = applyView(view);
  clear(table);

  const thead = table.createTHead();

  // A group row above the sortable one, when any column claims a group: one
  // label per config, spanning that config's columns.
  const grouped = view.columns.some((column) => column.group);
  if (grouped) {
    const groupRow = thead.insertRow();
    groupRow.className = 'group-head';
    let index = 0;
    while (index < view.columns.length) {
      const group = view.columns[index].group;
      const th = el('th');
      if (group) {
        // Consecutive columns of one config share a single spanning label.
        let span = 1;
        while (index + span < view.columns.length
               && view.columns[index + span].group === group) span += 1;
        th.textContent = group;
        th.colSpan = span;
        th.classList.add('group-start');
        index += span;
      } else {
        // An ungrouped column gets a blank placeholder, one per column, and its
        // real header in the row below. Spanning it down into that row instead
        // would consume a slot there too and shift every label sideways.
        index += 1;
      }
      groupRow.appendChild(th);
    }
  }

  const head = thead.insertRow();
  view.columns.forEach((column, index) => {
    const th = el('th', column.left ? 'left' : null);
    // The seam between one config's columns and the next.
    if (grouped && column.group
        && (index === 0 || view.columns[index - 1].group !== column.group)) {
      th.classList.add('group-start');
    }
    th.appendChild(document.createTextNode(column.title));
    // Headers double as the sort control: it is the affordance people reach for
    // before they look for a menu.
    th.classList.add('sortable');
    th.tabIndex = 0;
    th.setAttribute('role', 'button');
    const sorted = view.sort.key === column.key;
    if (sorted) {
      th.classList.add('is-sorted');
      th.setAttribute('aria-sort', view.sort.dir === 'asc' ? 'ascending' : 'descending');
    }
    th.appendChild(el('span', 'sort-caret', sorted && view.sort.dir === 'asc' ? '▲' : '▼'));
    th.title = 'Sort by ' + column.title;
    const toggle = () => {
      if (view.sort.key === column.key) {
        view.sort.dir = view.sort.dir === 'asc' ? 'desc' : 'asc';
      } else {
        view.sort.key = column.key;
        // Numbers are most useful largest-first; names read A-Z.
        view.sort.dir = column.type === 'num' ? 'desc' : 'asc';
      }
      paintTable(view);
      if (menuState.id === view.id) syncMenu();
    };
    th.addEventListener('click', toggle);
    th.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggle(); }
    });
    head.appendChild(th);
  });

  const body = table.createTBody();
  shown.forEach((row) => {
    const tr = body.insertRow();
    if (row.role === 'control') tr.className = 'is-context';
    view.columns.forEach((column, index) => {
      const cell = tr.insertCell();
      if (column.left) cell.className = 'left';
      if (grouped && column.group
          && (index === 0 || view.columns[index - 1].group !== column.group)) {
        cell.classList.add('group-start');
      }
      if (column.render) column.render(cell, row);
      else fillCell(cell, column.key, row);
    });
  });

  if (!shown.length && view.rows.length) {
    const cell = body.insertRow().insertCell();
    cell.colSpan = view.columns.length;
    const empty = el('div', 'empty');
    empty.appendChild(el('div', 'empty-glyph', '⊘'));
    empty.appendChild(el('div', 'empty-title', 'No rows match the filter'));
    empty.appendChild(el('div', 'empty-hint',
      'All ' + view.rows.length + ' row' + (view.rows.length === 1 ? '' : 's')
      + ' are still there — widen the filter or reset it to see them.'));
    cell.appendChild(empty);
  }

  updateViewChrome(view, shown.length);
}

/* The button and the note above the table both have to say when a filter is on.
 * A table that looks complete but is not would be this control's worst failure,
 * so it is stated twice. */
function updateViewChrome(view, shownCount) {
  const button = document.querySelector('[data-view="' + view.id + '"]');
  if (button) {
    const count = activeFilterCount(view);
    clear(button);
    button.appendChild(document.createTextNode('Filter & sort'));
    if (count) button.appendChild(el('span', 'btn-badge', String(count)));
    button.classList.toggle('is-filtering', count > 0);
  }

  const note = $(view.id + '-note');
  if (!note) return;
  const filtering = viewIsFiltering(view);
  const sortColumn = view.columns.find((c) => c.key === view.sort.key);
  clear(note);
  if (!filtering && !sortColumn) { note.hidden = true; return; }
  note.hidden = false;

  if (filtering) {
    const tag = el('span', 'tag is-info');
    tag.appendChild(document.createTextNode('filtered'));
    note.appendChild(tag);
    note.appendChild(document.createTextNode(
      'showing ' + shownCount + ' of ' + view.rows.length + ' rows'));
  }
  if (sortColumn) {
    note.appendChild(document.createTextNode(
      (filtering ? ' · ' : '') + 'sorted by ' + sortColumn.title + ' '
      + (view.sort.dir === 'asc' ? 'ascending' : 'descending')));
  }
  const reset = el('button', 'btn btn-sm', 'reset');
  reset.style.marginLeft = 'auto';
  reset.addEventListener('click', () => resetView(view.id));
  note.appendChild(reset);
}

function resetView(id) {
  Object.assign(tableViews[id], BLANK_VIEW());
  repaint(id);
  if (menuState.id === id) syncMenu();
}

/* --- the popover ---------------------------------------------------------- */

const menuState = { id: null, button: null };

/* Focus goes back to the button when the menu was dismissed from the keyboard
 * or by the button itself. After a click elsewhere it does not: that click has
 * already moved the user's attention, and yanking it back fights them. */
function closeMenu(restoreFocus) {
  const menu = $('view-menu');
  menu.hidden = true;
  if (menuState.button) {
    menuState.button.setAttribute('aria-expanded', 'false');
    if (restoreFocus) menuState.button.focus();
  }
  menuState.id = null;
  menuState.button = null;
}

function openMenu(id, button) {
  const menu = $('view-menu');
  menuState.id = id;
  menuState.button = button;
  button.setAttribute('aria-expanded', 'true');
  menu.hidden = false;
  syncMenu();

  // Measured from the button rather than an offsetParent, and flipped when it
  // would leave the viewport.
  const rect = button.getBoundingClientRect();
  const width = menu.offsetWidth;
  let left = rect.right - width;
  if (left < 8) left = 8;
  if (left + width > window.innerWidth - 8) left = window.innerWidth - width - 8;
  let top = rect.bottom + 6;
  const height = menu.offsetHeight;
  if (top + height > window.innerHeight - 8) top = Math.max(8, rect.top - height - 6);
  menu.style.left = left + 'px';
  menu.style.top = top + 'px';
  $('vm-text').focus();
}

/* The menu reads from whichever view is open; every control writes back and
 * repaints, so the table is never out of step with what the menu shows. */
function syncMenu() {
  const view = tableViews[menuState.id];
  if (!view) return;

  const sortSelect = $('vm-sort');
  clear(sortSelect);
  sortSelect.appendChild(new Option('— none —', ''));
  view.columns.forEach((column) => sortSelect.appendChild(new Option(column.title, column.key)));
  sortSelect.value = view.sort.key || '';

  document.querySelectorAll('#view-menu [data-dir]').forEach((button) =>
    button.setAttribute('aria-pressed', String(button.dataset.dir === view.sort.dir)));
  document.querySelectorAll('#view-menu [data-accuracy]').forEach((button) =>
    button.setAttribute('aria-pressed', String(button.dataset.accuracy === view.accuracy)));

  $('vm-text').value = view.text;

  // Only offer states this table actually contains, so the control cannot
  // filter everything away by naming something that is not there.
  const stateSelect = $('vm-state');
  const states = Array.from(new Set(view.rows.map((r) => r.status).filter(Boolean))).sort();
  clear(stateSelect);
  stateSelect.appendChild(new Option('any', ''));
  states.forEach((value) => stateSelect.appendChild(new Option(value, value)));
  stateSelect.value = states.includes(view.status) ? view.status : '';
  $('vm-state-row').hidden = states.length < 2;

  const numeric = view.columns.filter((c) => c.type === 'num');
  const numSelect = $('vm-num');
  clear(numSelect);
  numeric.forEach((column) => numSelect.appendChild(new Option(column.title, column.key)));
  if (!numeric.some((c) => c.key === view.numKey) && numeric.length) {
    view.numKey = numeric[0].key;
  }
  numSelect.value = view.numKey;
  $('vm-op').value = view.numOp;
  $('vm-value').value = view.numValue;

  // Accuracy is only a filter where some row has a verdict.
  $('vm-accuracy-row').hidden = !view.rows.some((r) => r.accuracy !== null && r.accuracy !== undefined);

  const shown = applyView(view).length;
  $('vm-count').textContent = shown === view.rows.length
    ? view.rows.length + ' rows'
    : shown + ' of ' + view.rows.length + ' rows';
}

/* History has its own painter; everything else is a results table. */
function repaint(id) {
  const view = tableViews[id];
  if (!view) return;
  if (id === 'history-table') paintHistory(view); else paintTable(view);
}

function commitMenu(mutate) {
  const view = tableViews[menuState.id];
  if (!view) return;
  mutate(view);
  repaint(view.id);
  syncMenu();
}

function setupTableMenu() {
  const menu = $('view-menu');

  document.querySelectorAll('[data-view]').forEach((button) => {
    button.addEventListener('click', (event) => {
      event.stopPropagation();
      const id = button.dataset.view;
      if (menuState.id === id) { closeMenu(true); return; }
      if (menuState.id) closeMenu(false);
      viewFor(id);
      openMenu(id, button);
    });
  });

  $('vm-sort').addEventListener('change', (e) =>
    commitMenu((view) => { view.sort.key = e.target.value || null; }));
  menu.querySelectorAll('[data-dir]').forEach((button) =>
    button.addEventListener('click', () =>
      commitMenu((view) => { view.sort.dir = button.dataset.dir; })));
  menu.querySelectorAll('[data-accuracy]').forEach((button) =>
    button.addEventListener('click', () =>
      commitMenu((view) => { view.accuracy = button.dataset.accuracy; })));
  $('vm-text').addEventListener('input', (e) =>
    commitMenu((view) => { view.text = e.target.value.trim(); }));
  $('vm-state').addEventListener('change', (e) =>
    commitMenu((view) => { view.status = e.target.value; }));
  $('vm-num').addEventListener('change', (e) =>
    commitMenu((view) => { view.numKey = e.target.value; }));
  $('vm-op').addEventListener('change', (e) =>
    commitMenu((view) => { view.numOp = e.target.value; }));
  $('vm-value').addEventListener('input', (e) =>
    commitMenu((view) => { view.numValue = e.target.value; }));
  $('vm-reset').addEventListener('click', () => { if (menuState.id) resetView(menuState.id); });

  menu.addEventListener('click', (event) => event.stopPropagation());
  document.addEventListener('click', () => { if (menuState.id) closeMenu(); });
  window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && menuState.id) { event.preventDefault(); closeMenu(true); }
  });
  // Reopening is cheaper than tracking the anchor through a scroll.
  window.addEventListener('resize', () => { if (menuState.id) closeMenu(); });
  document.querySelectorAll('.view').forEach((view) =>
    view.addEventListener('scroll', () => { if (menuState.id) closeMenu(); }, true));
}

/* Copies the table as shown — filtered and sorted — because that is what the
 * reader is looking at. The empty-state row is chrome, not data. */
function tableToCSV(table) {
  return Array.from(table.rows)
    .filter((row) => !row.querySelector('.empty'))
    .map((row) =>
    // A group header spans several columns in one cell; pad it out so every
    // line carries the same field count and the file opens straight in a sheet.
    Array.from(row.cells).reduce((fields, cell) => {
      const text = cell.textContent.replace(/"/g, '""');
      fields.push(/[",\n]/.test(text) ? '"' + text + '"' : text);
      for (let i = 1; i < (cell.colSpan || 1); i += 1) fields.push('');
      return fields;
    }, []).join(',')).join('\n');
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
/* `labels` renames the legend for a chart that is not measuring the baseline --
 * the compare view's bars are B against A. */
function renderBars(container, rows, labels) {
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
  const names = labels || { win: 'faster than the baseline',
                           loss: 'slower than the baseline' };
  legend.appendChild(item('is-mark', names.win));
  if (anyLoss) legend.appendChild(item('is-loss', names.loss));
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
  state.uis[key] = ui;
  clear(ui.log);
  ui.status.textContent = 'queued';
  ui.status.className = 'status-line is-running';
  ui.go.disabled = true;
  ui.stop.disabled = false;

  let since = 0;
  let consecutiveFailures = 0;

  const tick = async () => {
    let data;
    try {
      ({ data } = await api('/api/job/' + jobId + '?since=' + since));
    } catch (err) {
      // A dropped fetch is not the end of the job. Keep polling, but give up
      // after a run of failures rather than spinning forever against a server
      // that has gone away -- and hand the controls back when we do.
      if (++consecutiveFailures >= 5) {
        stopPolling(key);
        releaseControls(ui, 'lost contact with the server', 'failed');
        appendLog(ui.log, ['[dashboard] lost contact with the server; the '
          + 'benchmark may still be running. Reload to reattach.'], true);
      }
      return;
    }
    consecutiveFailures = 0;

    if (!data || data.error) {
      // The job is gone from the server's memory; nothing more is coming.
      stopPolling(key);
      releaseControls(ui, 'job no longer available', 'failed');
      return;
    }

    const finished = ['done', 'failed', 'stopped'].includes(data.status);

    // Settle the controls BEFORE rendering. Rendering is the elaborate part and
    // the part most likely to break in a future edit; if it throws, the buttons
    // must already be in the right state rather than stranded disabled.
    if (finished) {
      stopPolling(key);
      ui.go.disabled = false;
      ui.stop.disabled = true;
    }
    const settled = data.steps.filter((s) => ['done', 'failed', 'stopped'].includes(s.status)).length;
    const suffix = data.duration_s !== null ? ' · ' + data.duration_s + 's' : '';
    ui.status.textContent = finished
      ? data.status + suffix
      : data.status + ' · step ' + Math.min(settled + 1, data.steps.length)
        + '/' + data.steps.length + suffix;
    ui.status.className = 'status-line is-' + data.status;

    try {
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
    } catch (err) {
      // Report it rather than swallowing it: a table that silently stopped
      // updating is worse than one that says why.
      appendLog(ui.log, ['[dashboard] could not render this update: ' + err], true);
      if (window.console) console.error('render failed', err);
    }

    if (finished) {
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

/* Hand the controls back. Used wherever a job stops being watched, so there is
 * exactly one place that decides what "not running" looks like. */
function releaseControls(ui, message, tone) {
  ui.go.disabled = false;
  ui.stop.disabled = true;
  if (message) {
    ui.status.textContent = message;
    ui.status.className = 'status-line' + (tone ? ' is-' + tone : '');
  }
}

/* The safety net.
 *
 * Run is disabled while a job is watched and re-enabled when the poll sees it
 * finish. That chain has a single point of failure: if the poll ever stops --
 * a dropped connection, a bug in a render, a tab suspended by the browser --
 * the button stays disabled and the only way out is a reload.
 *
 * So the queue poll, which runs anyway, reconciles it: a view with no live poll
 * has nothing to wait for, and its Run button is enabled no matter how it got
 * into that state. */
function reconcileControls() {
  Object.keys(state.uis).forEach((key) => {
    const ui = state.uis[key];
    if (!ui || state.polls[key]) return;
    if (ui.go.disabled) {
      ui.go.disabled = false;
      ui.stop.disabled = true;
    }
  });
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
    // Nothing was running, so no poll is going to come along and re-enable Run.
    stopPolling(key);
    releaseControls(ui, 'already finished');
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

function mergedRunForm(form, shape) {
  if (!shape) return form;
  const merged = Object.assign({}, form);
  state.spec.shape_keys.forEach((key) => {
    if (shape[key] !== undefined) merged[key] = shape[key];
  });
  return merged;
}

function setupRun() {
  const form = state.forms.run;

  const updateEnvBadge = () => {
    const count = Object.keys(form.env || {}).length;
    const badge = $('run-env-count');
    badge.textContent = count + ' set';
    badge.hidden = !count;
  };
  // Declared before the toggle sets it, and read lazily, so the preview follows
  // whichever shape mode is active.
  let shapeForPreview = () => null;
  const preflight = makePreflight(
    () => mergedRunForm(form, shapeForPreview()),
    $('run-command'), $('run-issues'), $('run-derived'), updateEnvBadge);
  const changed = () => preflight();

  ['shape', 'optimization', 'timing', 'accuracy', 'data', 'torch', 'other']
    .forEach((group) => renderGroup($('run-' + group), group, form, changed));
  renderEnv($('run-env'), form, changed);

  fillPresetSelect($('run-preset'), (preset) => applyPreset($('run-shape'), preset));
  wireSpeedButtons();

  let mode = 'single';
  let picker = null;

  /* In many-shape mode each shape has its own command line, so the preview
   * shows the first ticked one rather than the typed fields it is not using. */
  const previewShape = () => (mode === 'many' && picker ? picker.selected()[0] : null);
  shapeForPreview = previewShape;

  const restate = () => {
    const chosen = (mode === 'many' && picker) ? picker.selected().length : 1;
    const note = $('run-shape-note');
    if (!note) return;
    note.textContent = mode === 'many'
      ? (chosen ? chosen + ' shape' + (chosen === 1 ? '' : 's') + ', one run each'
        : 'nothing ticked')
      : '';
  };

  picker = makeShapePicker($('run-shapes'), $('run-count'), $('run-all'),
    $('run-none'), () => { restate(); changed(); });
  $('run-presets-path').textContent = presetsFileName();

  const panes = $('panel-run').querySelectorAll('[data-run-pane]');
  const setMode = (next) => {
    mode = next;
    panes.forEach((pane) => { pane.hidden = pane.dataset.runPane !== next; });
    $('run-shape-mode').querySelectorAll('button').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.shapeMode === next));
    });
    try { localStorage.setItem('run-shape-mode', next); } catch (err) { /* no store */ }
    restate();
    changed();
  };
  $('run-shape-mode').addEventListener('click', (event) => {
    const button = event.target.closest('[data-shape-mode]');
    if (button) setMode(button.dataset.shapeMode);
  });

  let saved = null;
  try { saved = localStorage.getItem('run-shape-mode'); } catch (err) { saved = null; }
  setMode(saved === 'many' ? 'many' : 'single');

  $('run-copy-cmd').addEventListener('click', () => {
    navigator.clipboard.writeText($('run-command').textContent);
  });

  const ui = {
    log: $('run-log'), status: $('run-job-status'), go: $('run-go'), stop: $('run-stop'),
    follow: $('run-follow'), issues: $('run-issues'), resultsCard: $('run-results-card'),
    render: (rows) => { renderSummary($('run-summary'), rows); renderTable($('run-results'), rows); },
  };

  state.uis['run'] = ui;
  restate();
  $('run-go').addEventListener('click', () => {
    if (mode !== 'many') {
      submit('run', { mode: 'single', form }, ui);
      return;
    }
    const shapes = picker.selected();
    if (!shapes.length) {
      renderIssues($('run-issues'), [{ level: 'error', message: 'no shapes ticked' }]);
      return;
    }
    // The server already knows how to queue this: one child per shape, run one
    // at a time. It is still a sweep, and the history it writes still says so.
    submit('run', { mode: 'sweep', form, shapes }, ui);
  });
  $('run-stop').addEventListener('click', () => requestStop('run', ui));
  preflight();
}

/* ---------------------------------------------------------- compare view -- */

function mergedCompareForm(side, shape) {
  const merged = Object.assign({}, state.forms.cmpShape, side, { env: side.env || {} });
  // A ticked preset overrides the typed shape fields, exactly as the server
  // merges it on submit, so the preview shows what will actually run.
  if (shape) {
    state.spec.shape_keys.forEach((key) => {
      if (shape[key] !== undefined) merged[key] = shape[key];
    });
  }
  return merged;
}

function setupCompare() {
  const a = state.forms.cmpA;
  const b = state.forms.cmpB;

  let mode = 'single';
  let picker = null;

  /* In many-shape mode every shape has its own command line, so the preview
   * shows the first ticked one. A preview quietly showing the typed shape while
   * a different one runs would be worse than no preview at all. */
  const previewShape = () => (mode === 'many' && picker ? picker.selected()[0] : null);

  /* Say which shape the preview is of. Without this the command reads as the
   * whole run rather than the first of several. */
  const captionPreview = () => {
    const shape = previewShape();
    const note = (mode === 'many' && shape)
      ? 'the command for ' + shape.name + '; every other ticked shape runs the '
        + 'same flags with its own dimensions'
      : '';
    ['cmp-a-preview-note', 'cmp-b-preview-note'].forEach((id) => {
      const node = $(id);
      if (!node) return;
      node.textContent = note;
      node.hidden = !note;
    });
  };

  // Both sides are checked live and their issues merged, prefixed the way the
  // server prefixes them on submit - otherwise a setting that only breaks B
  // stays invisible until the run is rejected.
  const sideIssues = { A: [], B: [] };
  const showIssues = () => renderIssues($('cmp-issues'),
    sideIssues.A.map((i) => Object.assign({}, i, { message: 'A: ' + i.message }))
      .concat(sideIssues.B.map((i) => Object.assign({}, i, { message: 'B: ' + i.message }))));

  const preA = makePreflight(() => mergedCompareForm(a, previewShape()),
    $('cmp-a-command'), null, $('cmp-derived'),
    (data) => { sideIssues.A = data.issues || []; showIssues(); });
  const preB = makePreflight(() => mergedCompareForm(b, previewShape()),
    $('cmp-b-command'), null, null,
    (data) => { sideIssues.B = data.issues || []; showIssues(); });
  const both = () => { captionPreview(); preA(); preB(); };

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

  /* What is about to be queued. At two or three runs per shape the cost stops
   * being obvious, so it is stated beside the button rather than discovered in
   * the log. */
  const restate = () => {
    const perShape = $('cmp-control').checked ? 3 : 2;
    const shapes = (mode === 'many' && picker) ? picker.selected().length : 1;
    const total = shapes * perShape;
    $('cmp-run-count').textContent = shapes === 0
      ? 'no shapes ticked'
      : (mode === 'many' ? shapes + ' shapes × ' + perShape + ' = ' : '')
        + total + ' run' + (total === 1 ? '' : 's');
    $('cmp-control-scope').textContent = mode === 'many' ? ', on every shape' : '';
  };

  picker = makeShapePicker($('cmp-shapes'), $('cmp-count'), $('cmp-all'), $('cmp-none'),
    () => { restate(); both(); });
  $('cmp-presets-path').textContent = presetsFileName();

  const panes = $('panel-compare').querySelectorAll('[data-shape-pane]');
  const setMode = (next) => {
    mode = next;
    panes.forEach((pane) => { pane.hidden = pane.dataset.shapePane !== next; });
    $('cmp-shape-mode').querySelectorAll('button').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.shapeMode === next));
    });
    try { localStorage.setItem('cmp-shape-mode', next); } catch (err) { /* no store */ }
    restate();
    both();
  };
  $('cmp-shape-mode').addEventListener('click', (event) => {
    const button = event.target.closest('[data-shape-mode]');
    if (button) setMode(button.dataset.shapeMode);
  });
  $('cmp-control').addEventListener('change', restate);

  let saved = null;
  try { saved = localStorage.getItem('cmp-shape-mode'); } catch (err) { saved = null; }
  setMode(saved === 'many' ? 'many' : 'single');

  const ui = {
    log: $('cmp-log'), status: $('cmp-job-status'), go: $('cmp-go'), stop: $('cmp-stop'),
    follow: $('cmp-follow'), issues: $('cmp-issues'), resultsCard: $('cmp-results-card'),
    render: renderCompareResults,
  };

  state.uis['compare'] = ui;
  $('cmp-go').addEventListener('click', () => {
    const payload = {
      mode: 'compare',
      form_a: mergedCompareForm(a),
      form_b: mergedCompareForm(b),
      control: $('cmp-control').checked,
    };
    if (mode === 'many') {
      const shapes = picker.selected();
      if (!shapes.length) {
        renderIssues($('cmp-issues'), [{ level: 'error', message: 'no shapes ticked' }]);
        return;
      }
      payload.shapes = shapes;
    }
    submit('compare', payload, ui);
  });
  $('cmp-stop').addEventListener('click', () => requestStop('compare', ui));
  both();
}

/* --- the pivot: a row is a shape, a column group is a config --------------- */

/* Flat step rows in, one row per shape out, each holding its A, its B and the
 * control that decides whether the gap between them is a result. One-shape mode
 * is the same thing with a single group. */
function pivotCompareRows(rows) {
  const order = [];
  const groups = {};
  rows.forEach((row) => {
    const key = row.preset || '';
    if (!groups[key]) { groups[key] = []; order.push(key); }
    groups[key].push(row);
  });

  // The row's state is the least finished of its runs: a shape with B still
  // running is not "done" just because A is.
  const overall = (members) => {
    if (members.some((row) => row.status === 'failed')) return 'failed';
    if (members.some((row) => row.status === 'running')) return 'running';
    if (members.some((row) => row.status === 'stopped')) return 'stopped';
    if (members.every((row) => row.status === 'done')) return 'done';
    return 'queued';
  };

  return order.map((key) => {
    const members = groups[key];
    const side = (role) => members.find((row) => row.role === role) || null;
    const a = side('A');
    const b = side('B');
    const control = side('control');

    // Median latency, the same definition the single-shape view has always used.
    const ratio = (a && b && a.optimized && b.optimized)
      ? a.optimized / b.optimized : null;
    const noise = (a && control && a.optimized && control.optimized)
      ? Math.abs(a.optimized / control.optimized - 1) * 100 : null;
    const gap = ratio === null ? null : Math.abs(ratio - 1) * 100;

    let call = '';
    if (ratio !== null) {
      if (noise === null) call = 'no control';
      else if (gap <= noise) call = 'noise';
      else if (gap < noise * 2) call = 'marginal';
      else call = ratio > 1 ? 'B faster' : 'A faster';
    }

    // Either side failing accuracy makes the row's ratio meaningless, so a row
    // is never marked passing on the strength of the other half.
    const judged = [a, b].filter((row) => row && row.accuracy !== null);
    const accuracy = judged.length ? judged.every((row) => row.accuracy) : null;

    const seconds = members.reduce((sum, row) => sum + (row.secs || 0), 0);

    return {
      label: key || (a && a.shape) || (a && a.label) || '',
      a: a, b: b, control: control,
      ratio: ratio, noise: noise, gap: gap, call: call,
      // renderBars and the numeric filter both read `speedup`; here it is the
      // B-against-A ratio, which is what this table is about.
      speedup: ratio,
      accuracy: accuracy,
      status: overall(members),
      secs: seconds || null,
      errors: members.reduce((all, row) => all.concat(row.errors || []), []),
    };
  });
}

/* One config's column under a group header. The cell is drawn by the same
 * fillCell the other tables use, so "median ms" formats identically here --
 * including the /slice suffix on a streamed shape. */
function compareSideColumn(side, key, group, title, type) {
  return {
    key: side + key.charAt(0).toUpperCase() + key.slice(1),
    group: group,
    title: title,
    type: type,
    value: (row) => (row[side] ? cellValue({ key: key }, row[side]) : null),
    render: (cell, row) => {
      if (!row[side]) { cell.textContent = '–'; cell.classList.add('is-muted'); return; }
      fillCell(cell, key, row[side]);
    },
  };
}

const CALL_TONE = {
  'B faster': 'is-good',
  'A faster': 'is-critical',
  'noise': 'is-warning',
  'marginal': 'is-warning',
  'no control': 'is-warning',
};

const CALL_HELP = {
  'B faster': 'The gap is more than twice the noise floor measured on this shape.',
  'A faster': 'The gap is more than twice the noise floor measured on this shape.',
  'noise': 'The A-vs-B gap is inside this shape’s own control, so this run does '
    + 'not separate the two configs.',
  'marginal': 'Clears the control, but by less than 2x. Worth repeating before '
    + 'trusting it.',
  'no control': 'No control ran on this shape, so there is nothing to say whether '
    + 'the gap is real.',
};

const COMPARE_COLUMNS = [
  { key: 'label', title: 'shape', left: true, type: 'text' },
  { key: 'status', title: 'state', type: 'text' },
  compareSideColumn('a', 'optimized', 'Config A', 'median ms', 'num'),
  compareSideColumn('a', 'speedup', 'Config A', 'vs base', 'num'),
  compareSideColumn('a', 'accuracy', 'Config A', 'accuracy', 'text'),
  compareSideColumn('b', 'optimized', 'Config B', 'median ms', 'num'),
  compareSideColumn('b', 'speedup', 'Config B', 'vs base', 'num'),
  compareSideColumn('b', 'accuracy', 'Config B', 'accuracy', 'text'),
  {
    key: 'ratio', title: 'B vs A', type: 'num',
    value: (row) => row.ratio,
    render: (cell, row) => {
      if (row.ratio === null) { cell.textContent = '–'; cell.classList.add('is-muted'); return; }
      cell.textContent = row.ratio.toFixed(3) + 'x';
      cell.classList.add('is-strong', row.ratio >= 1 ? 'is-good' : 'is-critical');
      cell.title = 'A median / B median. Above 1.000x means B is the faster of '
        + 'the two on this shape.';
    },
  },
  {
    key: 'noise', title: 'control', type: 'num',
    value: (row) => row.noise,
    render: (cell, row) => {
      if (row.noise === null) { cell.textContent = '–'; cell.classList.add('is-muted'); return; }
      cell.textContent = row.noise.toFixed(1) + '%';
      cell.classList.add('is-muted');
      cell.title = 'Config A run twice on this shape. Its true ratio is 1.000x, so '
        + 'this is how far off a repeat of the same code lands — the bar the '
        + 'A-vs-B gap has to clear.';
    },
  },
  {
    key: 'call', title: 'verdict', type: 'text',
    value: (row) => row.call,
    render: (cell, row) => {
      if (!row.call) { cell.textContent = '–'; cell.classList.add('is-muted'); return; }
      const tone = CALL_TONE[row.call] || 'is-warning';
      const tag = el('span', 'tag ' + tone);
      tag.appendChild(el('span', null,
        tone === 'is-good' ? '✓' : (tone === 'is-critical' ? '✖' : '⚠')));
      tag.appendChild(document.createTextNode(row.call));
      cell.appendChild(tag);
      cell.title = CALL_HELP[row.call] || '';
    },
  },
  { key: 'secs', title: 'took', type: 'num' },
];

function renderCompareResults(rows) {
  const pivot = pivotCompareRows(rows);
  renderTable($('cmp-results'), pivot, null, COMPARE_COLUMNS);

  const summary = $('cmp-summary');
  clear(summary);
  if (pivot.length === 1) renderCompareOne(summary, pivot[0]);
  else if (pivot.length > 1) renderCompareMany(summary, pivot);
}

/* B against A on one shape, with the control's own deviation from 1.000x as the
 * bar the difference has to clear before it is called a result. */
function renderCompareOne(summary, row) {
  if (!row.a || !row.b || !row.a.optimized || !row.b.optimized) return;
  const ratio = row.ratio;
  const gapPct = row.gap;

  const headline = el('div', 'headline');
  headline.appendChild(heroBlock(ratio.toFixed(3) + 'x', 'B relative to A'));
  const tiles = [
    [fmt(row.a.optimized, 3) + ' ms', 'A median'],
    [fmt(row.b.optimized, 3) + ' ms', 'B median'],
  ];
  if (row.control && row.control.optimized) {
    tiles.push([fmt(row.control.optimized, 3) + ' ms', 'control median']);
  }
  headline.appendChild(tileRow(tiles));
  summary.appendChild(headline);

  const verdict = el('div', 'verdict');
  verdict.style.marginTop = '18px';

  if (row.noise !== null) {
    const controlRatio = row.a.optimized / row.control.optimized;
    const line = el('div');
    line.innerHTML = 'B is <b>' + ratio.toFixed(3) + 'x</b> config A, a gap of '
      + gapPct.toFixed(1) + '%. The control &mdash; A run twice, true value 1.000x '
      + '&mdash; came back at <b>' + controlRatio.toFixed(3) + 'x</b>, so the noise '
      + 'floor on this machine right now is ' + row.noise.toFixed(1) + '%.';
    verdict.appendChild(line);

    const call = el('div', 'v-call');
    let tone, glyph, text;
    if (gapPct <= row.noise) {
      tone = 'is-warning'; glyph = '⚠';
      text = 'Inside the noise. This run does not separate A from B — raise '
        + '--benchmark-rounds or --repeats, or accept that they measure the same.';
    } else if (gapPct < row.noise * 2) {
      tone = 'is-warning'; glyph = '⚠';
      text = 'Clears the control, but by less than 2x. Worth repeating before trusting.';
    } else {
      // ratio is A/B, so above 1.000x is B finishing sooner.
      tone = ratio > 1 ? 'is-good' : 'is-critical';
      glyph = ratio > 1 ? '✓' : '✖';
      text = ratio > 1
        ? 'B is the faster of the two, by comfortably more than the noise floor.'
        : 'B is the slower of the two, by comfortably more than the noise floor.';
    }
    const tag = el('span', 'tag ' + tone);
    tag.appendChild(el('span', null, glyph));
    tag.appendChild(document.createTextNode(gapPct <= row.noise ? 'not a result' : 'real'));
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

/* Many shapes: the headline is the geometric mean of the per-shape ratios, and
 * the count of rows that actually cleared their own control -- a mean over rows
 * that are mostly noise is a number with nothing behind it. */
function renderCompareMany(summary, pivot) {
  const scored = pivot.filter((row) => typeof row.ratio === 'number');
  if (!scored.length) return;

  const mean = geomean(scored.map((row) => row.ratio));
  const best = scored.reduce((x, y) => (x.ratio > y.ratio ? x : y));
  const worst = scored.reduce((x, y) => (x.ratio < y.ratio ? x : y));
  const real = scored.filter((row) => row.call === 'B faster' || row.call === 'A faster');
  const wins = real.filter((row) => row.ratio > 1).length;
  const losses = real.length - wins;
  const inside = scored.length - real.length;

  const headline = el('div', 'headline');
  headline.appendChild(heroBlock(mean.toFixed(3) + 'x',
    'B relative to A, geometric mean over ' + scored.length + ' shape'
      + (scored.length === 1 ? '' : 's'),
    mean >= 1 ? 'good' : 'critical'));
  headline.appendChild(tileRow([
    [best.ratio.toFixed(3) + 'x', 'best — ' + best.label],
    [worst.ratio.toFixed(3) + 'x', 'worst — ' + worst.label],
    [real.length + ' of ' + scored.length, 'cleared their control',
      'Rows whose A-vs-B gap is more than twice the noise floor measured on that '
      + 'same shape.'],
  ]));
  summary.appendChild(headline);

  const verdict = el('div', 'verdict');
  verdict.style.marginTop = '18px';
  const line = el('div');
  line.textContent = 'Every shape carries its own control, so every row is judged '
    + 'against the noise floor measured on that shape rather than a single figure '
    + 'borrowed across the set. ' + wins + ' came back faster on B, ' + losses
    + ' faster on A, and ' + inside + ' did not separate the two.';
  verdict.appendChild(line);

  if (inside === scored.length) {
    const call = el('div', 'v-call');
    const tag = el('span', 'tag is-warning');
    tag.appendChild(el('span', null, '⚠'));
    tag.appendChild(document.createTextNode('not a result'));
    call.appendChild(tag);
    call.appendChild(el('span', null, 'No shape separated the two configs. Raise '
      + '--benchmark-rounds or --repeats, or accept that they measure the same.'));
    verdict.appendChild(call);
  } else if (wins && losses) {
    const call = el('div', 'v-call');
    const tag = el('span', 'tag is-warning');
    tag.appendChild(el('span', null, '⚠'));
    tag.appendChild(document.createTextNode('shape dependent'));
    call.appendChild(tag);
    call.appendChild(el('span', null, 'B wins on some shapes and loses on others, '
      + 'so the geometric mean above hides a real reversal — read the rows.'));
    verdict.appendChild(call);
  }
  summary.appendChild(verdict);

  const judged = pivot.filter((row) => row.accuracy !== null);
  if (judged.length) {
    const failed = judged.filter((row) => row.accuracy === false);
    const strip = el('div', 'btn-row');
    strip.style.marginTop = '16px';
    const tag = el('span', 'tag ' + (failed.length ? 'is-critical' : 'is-good'));
    tag.appendChild(el('span', null, failed.length ? '✖' : '✓'));
    tag.appendChild(document.createTextNode(failed.length
      ? failed.length + ' of ' + judged.length + ' shapes failed accuracy'
      : 'accuracy passed on all ' + judged.length));
    strip.appendChild(tag);
    summary.appendChild(strip);
  }

  if (scored.length > 1) {
    const chart = el('div');
    chart.style.marginTop = '20px';
    // The caption goes in first: these bars are B against A, not the speedup
    // against the baseline that the same bars mean everywhere else in the app.
    const caption = el('div', 'prose');
    caption.style.marginBottom = '10px';
    caption.textContent = 'B relative to A per shape. The line at 1.000x is '
      + 'parity; bars past it are shapes where B finished sooner.';
    chart.appendChild(caption);
    if (renderBars(chart, scored, { win: 'B faster than A',
                                    loss: 'A faster than B' })) {
      summary.appendChild(chart);
    }
  }
}


/* The tickable shape list, shared by Run and Compare.
 *
 * `onChange` fires whenever the selection moves, so a caller can restate what
 * is about to be queued -- which matters more in Compare, where every ticked
 * shape costs two or three runs rather than one.
 */
/* The tail of the presets path, for the "edit presets.json" pointers. */
function presetsFileName() {
  return state.spec.presets_path.split(/[\\/]/).slice(-2).join('/');
}

function makeShapePicker(list, count, allButton, noneButton, onChange) {
  const build = () => {
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

  const selected = () => Array.from(list.querySelectorAll('input:checked'))
    .map((box) => state.spec.presets[Number(box.dataset.index)]);

  const setAll = (checked) => list.querySelectorAll('input')
    .forEach((box) => { if (!box.disabled) box.checked = checked; });

  const settle = () => {
    const chosen = list.querySelectorAll('input:checked').length;
    const blocked = state.spec.presets.filter((p) => p.blocked).length;
    if (count) {
      count.textContent = chosen === 0
        ? 'nothing selected'
        : chosen + ' shape' + (chosen === 1 ? '' : 's') + ' selected'
          + (blocked ? ' · ' + blocked + ' cannot run' : '');
    }
    if (onChange) onChange(chosen);
  };

  list.addEventListener('change', settle);
  if (allButton) allButton.addEventListener('click', () => { setAll(true); settle(); });
  if (noneButton) noneButton.addEventListener('click', () => { setAll(false); settle(); });

  build();
  settle();
  // Saving in the Presets view changes which shapes exist; a list still
  // offering the old ones would queue shapes no longer in the file.
  shapeLists.push(() => { build(); settle(); });

  return { selected, rebuild: () => { build(); settle(); } };
}

/* ---------------------------------------------------------- profile view -- */

/* Nanoseconds, at whatever scale reads. Every number nsys hands back is in ns,
 * and a table of nine-digit integers is unreadable. */
function ns(value) {
  if (value === null || value === undefined) return '–';
  const abs = Math.abs(value);
  if (abs < 1e3) return value.toFixed(0) + ' ns';
  if (abs < 1e6) return (value / 1e3).toFixed(2) + ' µs';
  if (abs < 1e9) return (value / 1e6).toFixed(3) + ' ms';
  return (value / 1e9).toFixed(3) + ' s';
}

/* C++ kernel signatures run to hundreds of characters. The part that
 * identifies the kernel is the template's own name, so the table shows that and
 * keeps the full signature on the row's tooltip. */
function shortKernel(name) {
  if (!name) return '';
  let text = String(name).replace(/^void\s+/, '').replace(/<unnamed>::/g, '');

  // Every CUTLASS GEMM arrives as cutlass::Kernel2<the_name_you_want>, so the
  // wrapper is the one thing the name does not tell you.
  const cutlass = text.match(/\b(cutlass_\w+)/);
  if (cutlass) return cutlass[1];

  let head = text.split(/[<(]/)[0]
    .replace(/^(at::native|at_cuda_detail::cub[\w:]*?|c10|thrust[\w:]*?)::/, '')
    .replace(/::$/, '');

  // ATen routes most elementwise work through two or three templates, so the
  // template alone names a dozen different kernels. The functor inside it is
  // what actually ran.
  const functor = text.match(
    /\b([A-Za-z_]\w*(?:Functor\w*|KernelImpl|Ops|CUDAKernel\w*))\b/);
  if (functor && head && functor[1] !== head && head.indexOf(functor[1]) === -1) {
    return head + ' · ' + functor[1];
  }
  return head || text.slice(0, 60);
}

/* Kernels this repository built, as opposed to ATen's and cuBLAS's. The NVTX
 * range already says which model issued a kernel; this says whose code it is,
 * which is the difference between "my attention kernel is the bottleneck" and
 * "the GEMM I do not control is". */
const OURS = /wmma_|tile_|fused_attention|fused_add_layernorm|warp_add_layernorm|gemm_bias_gelu|split_combine|identity_kernel/i;
const THEIRS = /^(void )?(at::|c10::|cutlass|ampere_|sm\d+_|cublas|gemv|splitKreduce)/;

function kernelOrigin(name) {
  if (THEIRS.test(name) && !OURS.test(name)) return 'library';
  return OURS.test(name) ? 'ours' : 'library';
}

const PROFILE_COLUMNS = [
  { key: 'share', title: 'share', type: 'num',
    value: (row) => row.share,
    render: (cell, row) => {
      // A proportion, drawn as a proportion. One measure, one hue.
      const wrap = el('div', 'share-cell');
      const track = el('div', 'share-track');
      const fill = el('div', 'share-fill');
      fill.style.width = (100 * row.share).toFixed(1) + '%';
      track.appendChild(fill);
      wrap.appendChild(track);
      wrap.appendChild(el('span', 'share-value', (100 * row.share).toFixed(1) + '%'));
      cell.appendChild(wrap);
    } },
  { key: 'name', title: 'kernel', left: true, type: 'text',
    value: (row) => row.short,
    render: (cell, row) => {
      const tag = el('span', 'tag ' + (row.origin === 'ours' ? 'is-info' : ''),
        row.origin === 'ours' ? 'ours' : 'library');
      tag.style.marginRight = '8px';
      cell.appendChild(tag);
      cell.appendChild(document.createTextNode(row.short));
      cell.title = row.name;
    } },
  { key: 'total', title: 'total', type: 'num',
    value: (row) => row.total_ns,
    render: (cell, row) => { cell.textContent = ns(row.total_ns); } },
  { key: 'instances', title: 'launches', type: 'num',
    value: (row) => row.instances,
    render: (cell, row) => { cell.textContent = row.instances.toLocaleString(); } },
  { key: 'avg', title: 'avg', type: 'num',
    value: (row) => row.avg_ns,
    render: (cell, row) => { cell.textContent = ns(row.avg_ns); } },
  { key: 'med', title: 'median', type: 'num',
    value: (row) => row.med_ns,
    render: (cell, row) => { cell.textContent = ns(row.med_ns); } },
  { key: 'max', title: 'max', type: 'num',
    value: (row) => row.max_ns,
    render: (cell, row) => { cell.textContent = ns(row.max_ns); } },
];

const MEMORY_COLUMNS = [
  { key: 'op', title: 'operation', left: true, type: 'text',
    value: (row) => row.op,
    render: (cell, row) => { cell.textContent = row.op; } },
  { key: 'count', title: 'count', type: 'num',
    value: (row) => row.count,
    render: (cell, row) => { cell.textContent = row.count; } },
  { key: 'total', title: 'total', type: 'num',
    value: (row) => row.total_ns,
    render: (cell, row) => { cell.textContent = ns(row.total_ns); } },
  { key: 'avg', title: 'avg', type: 'num',
    value: (row) => row.avg_ns,
    render: (cell, row) => { cell.textContent = ns(row.avg_ns); } },
];

function setupProfile() {
  const form = state.forms.profile;
  // The server's reduced iteration counts are the starting point, so the form
  // shows what will actually run rather than the harness's own defaults.
  Object.assign(form, state.spec.profile_defaults || {});

  const preflight = makePreflight(() => form, $('prof-command'), $('prof-issues'),
    $('prof-derived'));

  renderGroup($('prof-shape'), 'shape', form, preflight);
  renderGroup($('prof-timing'), 'timing', form, preflight);
  renderGroup($('prof-optimization'), 'optimization', form, preflight);
  renderEnv($('prof-env'), form, preflight);
  fillPresetSelect($('prof-preset'), (preset) => applyPreset($('prof-shape'), preset));

  // The reduced counts have to reach the fields too, or the form would show
  // the harness's defaults while the server quietly ran something else.
  Object.keys(state.spec.profile_defaults || {}).forEach((key) => {
    setFieldValue($('prof-timing'), key, state.spec.profile_defaults[key]);
  });

  renderTools($('prof-tools-body'), state.spec.tools || {});

  // Both routes to counter permission can look satisfied while collection still
  // fails, so the only trustworthy answer is to go and collect one.
  $('prof-ncu-probe').addEventListener('click', async () => {
    const button = $('prof-ncu-probe');
    button.disabled = true;
    const original = button.textContent;
    button.textContent = 'testing…';
    const { data } = await postJSON('/api/profile/ncu-probe', {});
    button.disabled = false;
    button.textContent = original;
    if (!data) return;
    if (data.error && !data.probed) {
      appendLog($('prof-log'), ['[dashboard] ' + data.error], true);
      return;
    }
    state.spec.tools = Object.assign({}, state.spec.tools, {
      ncu_counters_allowed: data.ok === true,
      ncu_reason: data.ok ? 'measured: ncu collected a counter'
                          : (data.error || 'ncu could not collect a counter'),
      ncu_measured: true,
    });
    renderTools($('prof-tools-body'), state.spec.tools);
  });

  const ui = {
    log: $('prof-log'), status: $('prof-job-status'), go: $('prof-go'),
    stop: $('prof-stop'), follow: $('prof-follow'), issues: $('prof-issues'),
    resultsCard: $('prof-results-card'),
    render: renderProfileResults,
  };

  state.uis['profile'] = ui;
  $('prof-devenv').addEventListener('change', () => {
    form.devenv = $('prof-devenv').checked;
  });
  $('prof-go').addEventListener('click', () => {
    form.devenv = $('prof-devenv').checked;
    submit('profile', { mode: 'profile', form }, ui);
  });
  $('prof-stop').addEventListener('click', () => requestStop('profile', ui));

  const ncuUi = {
    log: $('prof-log'), status: $('ncu-job-status'), go: $('ncu-go'),
    stop: $('ncu-stop'), follow: $('prof-follow'), issues: $('prof-issues'),
    resultsCard: $('ncu-results-card'),
    render: renderNcuResults,
  };
  state.uis['ncu'] = ncuUi;
  $('ncu-go').addEventListener('click', () => {
    form.devenv = $('prof-devenv').checked;
    form.launch_count = parseInt($('ncu-launches').value, 10) || 12;
    form.ncu_detail = $('ncu-detail').value;
    submit('ncu', { mode: 'ncu', form }, ncuUi);
  });
  $('ncu-stop').addEventListener('click', () => requestStop('ncu', ncuUi));

  $('prof-open').addEventListener('click', async () => {
    const { data } = await postJSON('/api/profile/open',
      { report: profileState.report });
    if (data && data.error) appendLog($('prof-log'), ['[dashboard] ' + data.error], true);
  });
  $('prof-delete').addEventListener('click', async () => {
    const { data } = await postJSON('/api/profile/delete',
      { report: profileState.report });
    if (data && data.removed) {
      appendLog($('prof-log'), ['[dashboard] removed ' + data.removed.join(', ')], true);
      $('prof-report-card').hidden = true;
    }
  });

  preflight();
}

/* What is installed and what it will let us do. The Nsight Compute row is the
 * point of this card: on a stock Windows box its counters are admin-only, and a
 * button that fails with ERR_NVGPUCTRPERM teaches nothing. */
function renderTools(container, tools) {
  clear(container);

  const row = (label, ok, detail, note) => {
    const line = el('div', 'tool-row');
    const tag = el('span', 'tag ' + (ok ? 'is-good' : 'is-warning'));
    tag.appendChild(el('span', null, ok ? '✓' : '⚠'));
    tag.appendChild(document.createTextNode(ok ? 'ready' : 'unavailable'));
    line.appendChild(tag);
    line.appendChild(el('span', 'tool-name', label));
    line.appendChild(el('span', 'tool-detail', detail || ''));
    container.appendChild(line);
    if (note) {
      const prose = el('p', 'prose');
      prose.style.margin = '6px 0 12px';
      prose.textContent = note;
      container.appendChild(prose);
    }
  };

  row('Nsight Systems', !!tools.nsys_available,
    tools.nsys ? tools.nsys.split(/[\\/]/).slice(-3, -1).join('/') : 'not found',
    tools.nsys_available ? '' : 'Install it, or set NSYS_PATH to nsys.exe.');

  // Until it has actually been measured, say so rather than guessing.
  const ncuReady = tools.ncu_available && tools.ncu_counters_allowed === true;
  const ncuUnknown = tools.ncu_available && !tools.ncu_measured
    && tools.ncu_counters_allowed !== true;
  const ncuDetail = tools.ncu
    ? tools.ncu.split(/[\\/]/).slice(-4, -3).join('/') : 'not found';
  if (ncuUnknown) {
    const line = el('div', 'tool-row');
    const tag = el('span', 'tag');
    tag.appendChild(el('span', null, '?'));
    tag.appendChild(document.createTextNode('not tested'));
    line.appendChild(tag);
    line.appendChild(el('span', 'tool-name', 'Nsight Compute'));
    line.appendChild(el('span', 'tool-detail', ncuDetail));
    container.appendChild(line);
    const note = el('p', 'prose');
    note.style.margin = '6px 0 12px';
    note.textContent = 'Whether counters can be collected cannot be read off the '
      + 'registry or from elevation: both can look satisfied while collection '
      + 'still fails, and a registry change does nothing until the driver '
      + 'reloads. Press "test counters" to find out. ' + (tools.ncu_reason || '');
    container.appendChild(note);
  } else {
    row('Nsight Compute', ncuReady, ncuDetail,
      ncuReady
        ? 'Measured: ncu collected a counter. Nothing here drives ncu yet, so '
          + 'this is detection only.'
        : (tools.ncu_available
          ? (tools.ncu_reason || 'counter collection failed') + ' '
            + (tools.ncu_fixes || []).join(' ')
          : 'Not found.'));
  }
  if (tools.elevated === true) {
    const note = el('p', 'prose');
    note.style.margin = '6px 0 0';
    note.textContent = 'This server is running elevated, so everything it '
      + 'launches — benchmarks included — runs elevated too.';
    container.appendChild(note);
  }
}

const profileState = { ranges: {}, range: '', report: '' };

function renderProfileResults(rows, data) {
  const step = (data.steps || []).find((s) => s.meta && s.meta.role === 'analyse');
  const result = (step && step.result) || null;
  profileState.report = (step && step.meta && step.meta.report) || '';
  if (!result) return;

  const analysis = result.analysis || {};
  profileState.ranges = analysis.ranges || {};

  renderProfileSummary($('prof-summary'), analysis);
  $('prof-results-card').hidden = false;

  renderRangePicker(analysis.ranges || {});
  renderProfileKernels();

  renderProfileMemory(result.tables || {}, result.skipped || {});
  renderProfileReport();
}

/* The headline: is this shape doing work, or waiting to be told to? */
function renderProfileSummary(container, analysis) {
  clear(container);
  const opt = analysis.optimized;
  const launch = analysis.launch_api;
  if (!opt) {
    const line = el('p', 'prose');
    line.textContent = 'No NVTX ranges were found in the trace. The harness only '
      + 'emits them when BENCH_NVTX=1, which the dashboard sets for a profile '
      + 'run — if this run came from somewhere else, that is why.';
    container.appendChild(line);
    return;
  }

  const busyPct = 100 * (opt.busy_fraction || 0);
  const headline = el('div', 'headline');
  headline.appendChild(heroBlock(busyPct.toFixed(1) + '%',
    'of one forward, the GPU is running a kernel',
    busyPct >= 70 ? 'good' : 'critical'));
  const tiles = [
    [ns(opt.busy_ns), 'busy per forward'],
    [ns(opt.idle_ns), 'idle per forward'],
    [String(opt.kernels), 'kernels per forward'],
    [ns(opt.largest_gap_ns), 'largest single gap'],
  ];
  if (analysis.custom_kernels && analysis.custom_kernels.count) {
    tiles.push([(100 * analysis.custom_kernels.share).toFixed(0) + '%',
      'in kernels from csrc/',
      'The share of the optimized model’s GPU time spent in kernels this '
      + 'repository built, as opposed to ATen, cuBLAS and CUTLASS. It is the '
      + 'ceiling on what optimizing your own code can win.']);
  }
  headline.appendChild(tileRow(tiles));
  container.appendChild(headline);

  // The check that caught a bad profile the first time this ran: the optimized
  // model was entirely ATen, cuBLAS and CUTLASS, because the extension had not
  // loaded and optimized/ fell back to SDPA without saying so. A breakdown of
  // somebody else's kernels presented as yours is worse than no breakdown.
  const custom = analysis.custom_kernels;
  if (custom && custom.count === 0) {
    const alarm = el('div', 'v-call');
    alarm.style.marginTop = '16px';
    const tag = el('span', 'tag is-critical');
    tag.appendChild(el('span', null, '✖'));
    tag.appendChild(document.createTextNode('no custom kernels ran'));
    alarm.appendChild(tag);
    alarm.appendChild(el('span', null,
      'Every kernel under "optimized" came from ATen, cuBLAS or CUTLASS, so '
      + 'this profile is of the fallback path, not of your code. Check the '
      + 'prepare step in the log: if the extension did not load, the harness '
      + 'runs SDPA instead and reports no error.'));
    container.appendChild(alarm);
  }

  $('prof-range-note').textContent = 'median of ' + opt.forwards
    + ' optimized forward' + (opt.forwards === 1 ? '' : 's');

  const verdict = el('div', 'verdict');
  verdict.style.marginTop = '18px';

  const line = el('div');
  line.innerHTML = 'Each figure is the <b>median</b> forward, not the total. The '
    + 'first forward a model runs pays for cuBLAS and module loading, and '
    + 'averaging that in would report the GPU as almost entirely idle.';
  verdict.appendChild(line);

  // The finding this whole view exists to make reproducible: a shape can be
  // slow because the kernels are slow, or because there are too many of them.
  const call = el('div', 'v-call');
  let tone, glyph, text;
  if (busyPct >= 80) {
    tone = 'is-good'; glyph = '✓';
    text = 'The GPU is busy nearly all of the forward, so the time is in the '
      + 'kernels themselves — the table below says which.';
  } else if (busyPct >= 50) {
    tone = 'is-warning'; glyph = '⚠';
    text = 'A fifth to a half of the forward is gaps between kernels. Fusing '
      + 'adjacent kernels or capturing a CUDA graph buys back time that no '
      + 'kernel optimization can.';
  } else {
    tone = 'is-critical'; glyph = '✖';
    text = 'Most of the forward is the GPU waiting. This shape is launch-bound, '
      + 'not compute-bound: making any single kernel faster will barely move it.';
  }
  const tag = el('span', 'tag ' + tone);
  tag.appendChild(el('span', null, glyph));
  tag.appendChild(document.createTextNode(
    busyPct >= 80 ? 'kernel-bound' : (busyPct >= 50 ? 'partly launch-bound'
                                                    : 'launch-bound')));
  call.appendChild(tag);
  call.appendChild(el('span', null, text));
  verdict.appendChild(call);

  if (launch && launch.calls) {
    const api = el('div');
    api.style.marginTop = '10px';
    api.innerHTML = '<b>' + launch.calls.toLocaleString() + '</b> cudaLaunchKernel '
      + 'calls across the whole trace, ' + ns(launch.avg_ns) + ' of CPU each.';
    verdict.appendChild(api);
  }
  container.appendChild(verdict);
}

/* One button per NVTX range. The optimized model leads, because it is the one
 * being worked on; the baseline is there to compare against. */
function renderRangePicker(ranges) {
  const picker = $('prof-range-pick');
  clear(picker);
  const names = Object.keys(ranges).sort((a, b) => {
    const rank = (n) => (n === 'optimized' ? 0 : (n === 'baseline' ? 1 : 2));
    return rank(a) - rank(b) || a.localeCompare(b);
  });
  if (!names.length) { $('prof-kernels-card').hidden = true; return; }
  if (names.indexOf(profileState.range) === -1) profileState.range = names[0];

  names.forEach((name) => {
    const button = el('button', null, name);
    button.setAttribute('aria-pressed', String(name === profileState.range));
    button.addEventListener('click', () => {
      profileState.range = name;
      renderRangePicker(profileState.ranges);
      renderProfileKernels();
    });
    picker.appendChild(button);
  });
  $('prof-kernels-card').hidden = false;
}

function renderProfileKernels() {
  const bucket = profileState.ranges[profileState.range];
  if (!bucket) return;
  const rows = bucket.kernels.map((kernel) => Object.assign({}, kernel, {
    short: shortKernel(kernel.name),
    origin: kernelOrigin(kernel.name),
    // The filter's free-text box matches on label.
    label: shortKernel(kernel.name),
  }));
  renderTable($('prof-kernels'), rows, null, PROFILE_COLUMNS);
}

function renderProfileMemory(tables, skipped) {
  const rows = (tables.cuda_gpu_mem_time_sum || []).map((row) => ({
    op: row['Operation'] || row['Name'] || '',
    label: row['Operation'] || row['Name'] || '',
    count: Number(row['Count'] || row['Instances'] || 0),
    total_ns: Number(row['Total Time (ns)'] || 0),
    avg_ns: Number(row['Avg (ns)'] || 0),
  }));
  if (!rows.length) {
    // Usually correct rather than broken: the harness generates its input on
    // the device, so a clean run copies nothing.
    $('prof-memory-card').hidden = true;
    return;
  }
  $('prof-memory-card').hidden = false;
  renderTable($('prof-memory'), rows, null, MEMORY_COLUMNS);
}

async function renderProfileReport() {
  if (!profileState.report) { $('prof-report-card').hidden = true; return; }
  const name = profileState.report.split(/[\\/]/).pop();
  $('prof-report-path').textContent = profileState.report;
  $('prof-report-card').hidden = false;
  const { data } = await api('/api/profile/report?name=' + encodeURIComponent(name));
  $('prof-report-size').textContent = (data && data.exists)
    ? data.human + ' on disk' : 'not on disk';
}


/* --- Nsight Compute: why a kernel takes the time nsys says it takes -------- */

const VERDICT_TONE = {
  'compute bound': 'is-good',
  'memory bound': 'is-warning',
  'latency bound': 'is-critical',
  'leans compute': 'is-info',
  'leans memory': 'is-info',
  'balanced': 'is-info',
};

const pct = (value) => (value === null || value === undefined)
  ? '–' : value.toFixed(1) + '%';

/* Up to the first sentence end that is not a decimal point or an abbreviation:
 * these rules are full of "5.3 sectors" and "96.39% of", and cutting at every
 * period would stop at the first number. */
function firstSentence(text) {
  const flat = String(text || '').replace(/\s+/g, ' ').trim();
  const end = flat.search(/[.:](?=\s+[A-Z])/);
  return end > 0 ? flat.slice(0, end + 1) : flat;
}

/* A percentage of peak, drawn against its peak. The bar is the point: 90% and
 * 23% are the whole diagnosis, and two numbers side by side hide it. */
function peakCell(cell, value) {
  if (value === null || value === undefined) {
    cell.textContent = '–'; cell.classList.add('is-muted'); return;
  }
  const wrap = el('div', 'share-cell');
  const track = el('div', 'share-track');
  const fill = el('div', 'share-fill' + (value >= 80 ? ' is-hot' : ''));
  fill.style.width = Math.max(0, Math.min(100, value)).toFixed(1) + '%';
  track.appendChild(fill);
  wrap.appendChild(track);
  wrap.appendChild(el('span', 'share-value', value.toFixed(1) + '%'));
  cell.appendChild(wrap);
}

const NCU_COLUMNS = [
  { key: 'name', title: 'kernel', left: true, type: 'text',
    value: (row) => row.short,
    render: (cell, row) => {
      cell.textContent = row.short;
      cell.title = row.name;
    } },
  { key: 'duration', title: 'duration', type: 'num',
    value: (row) => row.duration_ns,
    render: (cell, row) => { cell.textContent = ns(row.duration_ns); } },
  { key: 'compute', title: 'compute', type: 'num',
    value: (row) => row.compute_pct,
    render: (cell, row) => peakCell(cell, row.compute_pct) },
  { key: 'memory', title: 'memory', type: 'num',
    value: (row) => row.memory_pct,
    render: (cell, row) => peakCell(cell, row.memory_pct) },
  { key: 'tensor', title: 'tensor', type: 'num',
    value: (row) => row.tensor_pct,
    render: (cell, row) => {
      // The whole point of the wmma kernels is that this is not near zero, so
      // it earns a column rather than a line in the details.
      peakCell(cell, row.tensor_pct);
      if (row.tensor_pct !== null && row.tensor_pct !== undefined) {
        cell.title = 'Share of peak that the tensor pipe was active. A kernel '
          + 'that should be using tensor cores and reads near zero here is not '
          + 'using them.';
      }
    } },
  { key: 'occupancy', title: 'occupancy', type: 'num',
    value: (row) => row.occupancy_pct,
    render: (cell, row) => {
      if (row.occupancy_pct === null || row.occupancy_pct === undefined) {
        cell.textContent = '–'; cell.classList.add('is-muted'); return;
      }
      cell.textContent = pct(row.occupancy_pct);
      if (row.theoretical_occupancy_pct !== null
          && row.theoretical_occupancy_pct !== undefined) {
        cell.title = 'achieved ' + pct(row.occupancy_pct) + ' of a theoretical '
          + pct(row.theoretical_occupancy_pct)
          + (row.occupancy_limiter ? ', capped by ' + row.occupancy_limiter : '');
      }
      if (row.occupancy_limiter) cell.classList.add('is-warning');
    } },
  { key: 'registers', title: 'reg/thread', type: 'num',
    value: (row) => row.registers,
    render: (cell, row) => {
      cell.textContent = row.registers === null || row.registers === undefined
        ? '–' : String(row.registers);
    } },
  { key: 'verdict', title: 'limited by', type: 'text',
    value: (row) => row.verdict,
    render: (cell, row) => {
      if (!row.verdict) { cell.textContent = '–'; cell.classList.add('is-muted'); return; }
      const tag = el('span', 'tag ' + (VERDICT_TONE[row.verdict] || 'is-info'),
        row.verdict);
      cell.appendChild(tag);
      cell.title = row.why || '';
    } },
];

function renderNcuResults(rows, data) {
  const step = (data.steps || []).find((s) => s.meta && s.meta.role === 'collect');
  const result = (step && step.result) || null;
  if (!result) return;

  const blocked = (result.error_lines || [])
    .some((line) => line.indexOf('ERR_NVGPUCTRPERM') !== -1);
  const kernels = result.kernels || [];

  $('ncu-results-card').hidden = false;
  const level = (step.meta && step.meta.detail) || '';
  $('ncu-note').textContent = kernels.length
    ? kernels.length + ' launch' + (kernels.length === 1 ? '' : 'es') + ' profiled'
      + (level ? ' · ' + level : '')
    : '';

  renderTable($('ncu-results'), kernels.map((k) => Object.assign({}, k, {
    short: shortKernel(k.name),
    label: shortKernel(k.name),
  })), null, NCU_COLUMNS);

  const guide = $('ncu-guidance');
  clear(guide);

  if (blocked) {
    const call = el('div', 'v-call');
    const tag = el('span', 'tag is-critical');
    tag.appendChild(el('span', null, '✖'));
    tag.appendChild(document.createTextNode('counters refused'));
    call.appendChild(tag);
    call.appendChild(el('span', null,
      'The driver returned ERR_NVGPUCTRPERM, so nothing was collected. Counters '
      + 'are administrator-only until RmProfilingAdminOnly is 0 and the driver '
      + 'has reloaded — that means a reboot — or until the dashboard itself is '
      + 'started from an elevated terminal.'));
    guide.appendChild(call);
    return;
  }

  if (!kernels.length) {
    const line = el('p', 'prose');
    line.textContent = 'No kernel matched. Counters are collected only for '
      + 'kernels defined in csrc/, so this means the run used the fallback path '
      + '— the same thing the timeline reports as "no custom kernels ran".';
    guide.appendChild(line);
    return;
  }

  // The counters that have no column: memory traffic, coalescing, and whatever
  // else the chosen detail level collected.
  const withExtras = kernels.filter((k) => (k.extras || []).length);
  if (withExtras.length) {
    const heading = el('div', 'field-label');
    heading.style.margin = '4px 0 10px';
    heading.textContent = 'counters';
    guide.appendChild(heading);

    withExtras.forEach((kernel) => {
      const block = el('div');
      block.style.marginBottom = '12px';
      const name = el('div');
      name.style.font = '500 12px/1.5 var(--mono)';
      name.textContent = shortKernel(kernel.name);
      block.appendChild(name);

      const grid = el('div', 'counter-grid');
      kernel.extras.forEach((metric) => {
        const item = el('div', 'counter');
        item.appendChild(el('span', 'counter-label', metric.label));
        const shown = metric.value === null || metric.value === undefined
          ? (metric.text || '–')
          : (metric.unit === '%' ? metric.value.toFixed(1) + '%'
            : Math.round(metric.value).toLocaleString());
        item.appendChild(el('span', 'counter-value', shown));
        item.title = metric.key;
        grid.appendChild(item);
      });

      // Sectors per request is the coalescing number and reads as nothing
      // without its ideal beside it.
      if (kernel.sectors_per_request) {
        const item = el('div', 'counter');
        item.appendChild(el('span', 'counter-label', 'sectors per request'));
        const value = el('span', 'counter-value',
          kernel.sectors_per_request.toFixed(1) + ' / 4 ideal');
        if (kernel.sectors_per_request > 8) value.classList.add('is-critical');
        else if (kernel.sectors_per_request > 4.5) value.classList.add('is-warning');
        item.appendChild(value);
        item.title = 'Global load sectors divided by requests. 4 is fully '
          + 'coalesced; higher means each request is pulling more cache lines '
          + 'than it needs.';
        grid.appendChild(item);
      }
      block.appendChild(grid);
      guide.appendChild(block);
    });
  }

  // NVIDIA's own analysis, passed through rather than paraphrased. It is more
  // specific than anything this page could infer from the same numbers.
  const withRules = kernels.filter((k) => (k.rules || []).length);
  if (!withRules.length) return;
  const heading = el('div', 'field-label');
  heading.style.margin = '4px 0 10px';
  heading.textContent = 'what Nsight Compute says';
  guide.appendChild(heading);

  withRules.forEach((kernel) => {
    const block = el('div');
    block.style.marginBottom = '12px';
    const name = el('div');
    name.style.font = '500 12px/1.5 var(--mono)';
    name.textContent = shortKernel(kernel.name);
    block.appendChild(name);
    kernel.rules.slice(0, 3).forEach((rule) => {
      const line = el('div', 'prose rule-line');
      line.style.margin = '3px 0 0 12px';
      const tag = el('span', 'tag ' + (rule.type === 'OPT' ? 'is-warning' : ''),
        rule.type || 'INF');
      tag.style.marginRight = '8px';
      line.appendChild(tag);
      // The first sentence is the finding; what follows is how it was derived,
      // which is worth keeping but not worth a paragraph on screen.
      line.appendChild(document.createTextNode(firstSentence(rule.text)));
      line.title = rule.text;
      block.appendChild(line);
    });
    guide.appendChild(block);
  });
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

  state.uis['script'] = ui;
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
  ['run-preset', 'cmp-preset', 'prof-preset'].forEach((id) => {
    const select = $(id);
    if (!select) return;
    while (select.options.length > 1) select.remove(1);
    state.spec.presets.forEach((preset, index) => {
      const option = new Option(preset.name, String(index));
      if (preset.note) option.title = preset.note;
      select.appendChild(option);
    });
  });
  shapeLists.forEach((rebuild) => rebuild());
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

const HISTORY_COLUMNS = [
  { key: 'when', title: 'when', left: true, type: 'text' },
  { key: 'label', title: 'job', left: true, type: 'text' },
  { key: 'mode', title: 'mode', left: true, type: 'text' },
  { key: 'status', title: 'status', left: true, type: 'text' },
  { key: 'steps', title: 'steps', type: 'num' },
  { key: 'best', title: 'best speedup', type: 'num' },
  { key: 'secs', title: 'took', type: 'num' },
];

function paintHistory(view) {
  const table = $(view.id);
  const shown = applyView(view);
  clear(table);

  const head = table.createTHead().insertRow();
  view.columns.forEach((column) => {
    const th = el('th', column.left ? 'left' : null);
    th.appendChild(document.createTextNode(column.title));
    th.classList.add('sortable');
    th.tabIndex = 0;
    th.setAttribute('role', 'button');
    const sorted = view.sort.key === column.key;
    if (sorted) {
      th.classList.add('is-sorted');
      th.setAttribute('aria-sort', view.sort.dir === 'asc' ? 'ascending' : 'descending');
    }
    th.appendChild(el('span', 'sort-caret', sorted && view.sort.dir === 'asc' ? '▲' : '▼'));
    const toggle = () => {
      if (view.sort.key === column.key) view.sort.dir = view.sort.dir === 'asc' ? 'desc' : 'asc';
      else { view.sort.key = column.key; view.sort.dir = column.type === 'num' ? 'desc' : 'asc'; }
      paintHistory(view);
      if (menuState.id === view.id) syncMenu();
    };
    th.addEventListener('click', toggle);
    th.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggle(); }
    });
    head.appendChild(th);
  });

  const body = table.createTBody();
  if (!shown.length) {
    const cell = body.insertRow().insertCell();
    cell.colSpan = view.columns.length;
    const empty = el('div', 'empty');
    empty.appendChild(el('div', 'empty-glyph', view.rows.length ? '⊘' : '○'));
    empty.appendChild(el('div', 'empty-title',
      view.rows.length ? 'No rows match the filter' : 'No finished jobs yet'));
    empty.appendChild(el('div', 'empty-hint', view.rows.length
      ? 'All ' + view.rows.length + ' rows are still there — widen the filter or reset it.'
      : 'Runs are recorded here once they finish, with their full logs kept beside '
        + 'the index in dashboard/runs/.'));
    cell.appendChild(empty);
    updateViewChrome(view, 0);
    return;
  }

  shown.forEach((row) => {
    const tr = body.insertRow();
    tr.title = row.title || '';
    view.columns.forEach((column) => {
      const cell = tr.insertCell();
      if (column.left) cell.className = 'left';
      const value = row[column.key];
      if (column.key === 'best') {
        cell.textContent = value === null ? '–' : value.toFixed(3) + 'x';
        if (value !== null) cell.classList.add(value >= 1 ? 'is-good' : 'is-critical');
      } else if (column.key === 'secs') {
        cell.textContent = value === null || value === undefined ? '–' : value + 's';
      } else if (column.key === 'status') {
        cell.textContent = value;
        cell.classList.add(value === 'done' ? 'is-good'
          : (value === 'failed' ? 'is-critical' : 'is-muted'));
      } else {
        cell.textContent = value === null || value === undefined ? '–' : String(value);
      }
    });
  });
  updateViewChrome(view, shown.length);
}

async function refreshHistory() {
  const { data } = await api('/api/history?limit=60');
  const entries = (data && data.entries) || [];
  const view = viewFor('history-table');
  view.columns = HISTORY_COLUMNS;
  view.rows = entries.map((entry) => {
    const speedups = (entry.steps || [])
      .map((step) => step.result && step.result.speedup)
      .filter((value) => typeof value === 'number');
    return {
      when: new Date(entry.created_at * 1000).toLocaleString(),
      // `label` so the popover's name filter works the same way it does on the
      // results tables.
      label: entry.id,
      mode: entry.mode,
      status: entry.status,
      steps: (entry.steps || []).length,
      best: speedups.length ? Math.max.apply(null, speedups) : null,
      secs: entry.duration_s,
      title: entry.title || '',
      accuracy: null,
    };
  });
  paintHistory(view);
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
  reconcileControls();
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
  const order = ['run', 'compare', 'profile', 'scripts', 'presets', 'history'];
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
  setupTableMenu();
  setupRun();
  setupCompare();
  setupProfile();
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

import './style.css';
import {
  approvalArtifacts,
  loadRun,
  loadRuns,
  PipelineEventStream,
  requestApiToken,
  sendCommand,
  sendMessage,
} from './api.ts';
import { layoutDag } from './dag.ts';
import { redactSecrets } from './redact.ts';
import { dashboardReducer, initialState, selectedEntities } from './state.ts';
import type {
  Artifact,
  DashboardAction,
  DashboardState,
  PipelineCommand,
  PipelineRun,
  PipelineStage,
  StageAttempt,
} from './types.ts';

function element<K extends keyof HTMLElementTagNameMap>(
  name: K,
  className?: string,
  content?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(name);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}

function displayStatus(status: string): string {
  return status.replaceAll('_', ' ');
}

function statusPill(status: string): HTMLSpanElement {
  const pill = element('span', `status status--${status}`, displayStatus(status));
  return pill;
}

function displayTime(value?: string): string {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? value
    : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date);
}

function duration(attempt: StageAttempt): string {
  if (!attempt.startedAt) return 'Not started';
  const start = Date.parse(attempt.startedAt);
  const end = attempt.finishedAt ? Date.parse(attempt.finishedAt) : Date.now();
  if (!Number.isFinite(start) || !Number.isFinite(end)) return displayTime(attempt.startedAt);
  const seconds = Math.max(0, Math.round((end - start) / 1_000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function safeArtifactUrl(value?: string): string | undefined {
  if (!value) return undefined;
  try {
    const url = new URL(value, window.location.origin);
    if (
      url.origin !== window.location.origin ||
      url.username ||
      url.password ||
      !url.pathname.startsWith('/api/pipeline/')
    ) {
      return undefined;
    }
    return url.href;
  } catch {
    return undefined;
  }
}

class Dashboard {
  private state: DashboardState = initialState;
  private stream?: PipelineEventStream;
  private readonly runList = document.querySelector<HTMLElement>('#run-list')!;
  private readonly title = document.querySelector<HTMLElement>('#run-title')!;
  private readonly runMeta = document.querySelector<HTMLElement>('#run-meta')!;
  private readonly connection = document.querySelector<HTMLElement>('#connection')!;
  private readonly budget = document.querySelector<HTMLElement>('#budget')!;
  private readonly graph = document.querySelector<HTMLElement>('#dag')!;
  private readonly attempts = document.querySelector<HTMLElement>('#attempts')!;
  private readonly detail = document.querySelector<HTMLElement>('#attempt-detail')!;
  private readonly actions = document.querySelector<HTMLElement>('#actions')!;
  private readonly notice = document.querySelector<HTMLElement>('#notice')!;
  private readonly form = document.querySelector<HTMLFormElement>('#message-form')!;
  private readonly input = document.querySelector<HTMLTextAreaElement>('#message')!;
  private refreshTimer?: number;
  private pollTimer?: number;

  constructor() {
    this.form.addEventListener('submit', (event) => void this.submitMessage(event));
    window.addEventListener('beforeunload', () => {
      this.stream?.stop();
      window.clearInterval(this.pollTimer);
    });
  }

  async start(): Promise<void> {
    if (!requestApiToken()) {
      this.dispatch({ type: 'error', message: 'A pipeline access token is required' });
      this.dispatch({ type: 'connection', connection: 'offline' });
      return;
    }
    try {
      this.dispatch({ type: 'load', runs: await loadRuns() });
      this.connectSelectedRun();
      this.pollTimer = window.setInterval(() => {
        const runId = this.state.selectedRunId;
        if (runId) this.scheduleRefresh(runId);
      }, 3_000);
    } catch (error) {
      this.dispatch({
        type: 'error',
        message: error instanceof Error ? error.message : 'Unable to load pipeline runs',
      });
      this.dispatch({ type: 'connection', connection: 'offline' });
    }
  }

  private dispatch(action: DashboardAction): void {
    const oldRunId = this.state.selectedRunId;
    this.state = dashboardReducer(this.state, action);
    this.render();
    if (action.type === 'select-run' && oldRunId !== this.state.selectedRunId) {
      this.connectSelectedRun();
    }
  }

  private connectSelectedRun(): void {
    this.stream?.stop();
    const runId = this.state.selectedRunId;
    if (!runId) return;
    this.dispatch({ type: 'connection', connection: 'connecting' });
    this.stream = new PipelineEventStream(
      runId,
      {
        onConnection: (connection) => this.dispatch({ type: 'connection', connection }),
        onEvent: (event, eventId) => {
          this.dispatch({ type: 'event', event, eventId });
          if (event.type === 'refresh') this.scheduleRefresh(event.runId);
        },
        onError: (message) => this.dispatch({ type: 'error', message }),
      },
      this.state.lastEventId,
    );
    this.stream.start();
  }

  private scheduleRefresh(runId: string): void {
    window.clearTimeout(this.refreshTimer);
    this.refreshTimer = window.setTimeout(async () => {
      try {
        const refreshed = await loadRun(runId);
        this.dispatch({ type: 'event', event: { type: 'run.upsert', run: refreshed } });
      } catch (error) {
        this.dispatch({
          type: 'error',
          message: error instanceof Error ? error.message : 'Unable to refresh pipeline run',
        });
      }
    }, 100);
  }

  private render(): void {
    const { run, stage, attempt } = selectedEntities(this.state);
    this.renderRuns(run);
    this.renderHeader(run);
    this.renderDag(run, stage);
    this.renderAttempts(stage, attempt);
    this.renderDetail(attempt);
    this.renderActions(run, stage, attempt);
    this.notice.textContent = this.state.error ?? '';
    this.notice.hidden = !this.state.error;
    this.form.hidden = !run;
    this.input.disabled = !run;
  }

  private renderRuns(selected?: PipelineRun): void {
    this.runList.replaceChildren();
    if (!this.state.runs.length) {
      this.runList.append(element('p', 'empty', 'No pipeline runs yet.'));
      return;
    }
    for (const run of this.state.runs) {
      const button = element('button', 'run-card');
      button.type = 'button';
      button.classList.toggle('is-selected', run.id === selected?.id);
      button.setAttribute('aria-pressed', String(run.id === selected?.id));
      const top = element('span', 'run-card__top');
      top.append(element('strong', '', run.name), statusPill(run.status));
      button.append(
        top,
        element('span', 'run-card__time', displayTime(run.updatedAt ?? run.createdAt)),
      );
      button.addEventListener('click', () => this.dispatch({ type: 'select-run', runId: run.id }));
      this.runList.append(button);
    }
  }

  private renderHeader(run?: PipelineRun): void {
    this.title.textContent = run?.name ?? 'Pipeline dashboard';
    this.runMeta.replaceChildren();
    if (run) {
      this.runMeta.append(statusPill(run.status), element('span', '', `Run ${run.id}`));
      if (run.updatedAt)
        this.runMeta.append(element('span', '', `Updated ${displayTime(run.updatedAt)}`));
    }
    this.connection.className = `connection connection--${this.state.connection}`;
    this.connection.textContent = displayStatus(this.state.connection);
    this.budget.replaceChildren();
    if (!run?.budget) {
      this.budget.append(element('span', 'muted', 'No budget reported'));
      return;
    }
    const { spent, limit, currency } = run.budget;
    const ratio = limit > 0 ? Math.min(1, spent / limit) : 0;
    const label = `${spent.toFixed(2)} / ${limit.toFixed(2)} ${currency}`;
    const meter = element('progress');
    meter.max = Math.max(limit, 1);
    meter.value = spent;
    meter.setAttribute('aria-label', `Budget used: ${label}`);
    this.budget.append(element('span', 'budget__label', label), meter);
    this.budget.classList.toggle('budget--warning', ratio >= 0.8);
  }

  private renderDag(run?: PipelineRun, selected?: PipelineStage): void {
    this.graph.replaceChildren();
    if (!run?.stages.length) {
      this.graph.append(element('p', 'empty', 'No stages reported.'));
      return;
    }
    const layout = layoutDag(run.stages);
    const canvas = element('div', 'dag__canvas');
    canvas.style.width = `${layout.width}px`;
    canvas.style.height = `${layout.height}px`;
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', String(layout.width));
    svg.setAttribute('height', String(layout.height));
    svg.setAttribute('aria-hidden', 'true');
    const positions = new Map(layout.nodes.map((node) => [node.id, node]));
    for (const stage of run.stages) {
      const target = positions.get(stage.id);
      if (!target) continue;
      for (const parentId of stage.dependsOn) {
        const source = positions.get(parentId);
        if (!source) continue;
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const sx = source.x + 154;
        const sy = source.y + 25;
        const tx = target.x;
        const ty = target.y + 25;
        path.setAttribute('d', `M ${sx} ${sy} C ${sx + 28} ${sy}, ${tx - 28} ${ty}, ${tx} ${ty}`);
        svg.append(path);
      }
    }
    canvas.append(svg);
    for (const stage of run.stages) {
      const position = positions.get(stage.id)!;
      const node = element('button', `dag-node dag-node--${stage.status}`);
      node.type = 'button';
      node.style.left = `${position.x}px`;
      node.style.top = `${position.y}px`;
      node.classList.toggle('is-selected', stage.id === selected?.id);
      node.setAttribute('aria-pressed', String(stage.id === selected?.id));
      node.setAttribute('aria-label', `${stage.name}: ${displayStatus(stage.status)}`);
      node.append(
        element('span', 'dag-node__name', stage.name),
        element('span', 'dag-node__status', displayStatus(stage.status)),
      );
      node.addEventListener('click', () =>
        this.dispatch({ type: 'select-stage', stageId: stage.id }),
      );
      canvas.append(node);
    }
    this.graph.append(canvas);
  }

  private renderAttempts(stage?: PipelineStage, selected?: StageAttempt): void {
    this.attempts.replaceChildren();
    if (!stage) {
      this.attempts.append(element('p', 'empty', 'Select a stage.'));
      return;
    }
    const heading = element('div', 'section-heading');
    heading.append(element('h2', '', stage.name), statusPill(stage.status));
    this.attempts.append(heading);
    if (!stage.attempts.length) {
      this.attempts.append(element('p', 'empty', 'No attempts yet.'));
      return;
    }
    const list = element('div', 'attempt-list');
    [...stage.attempts].reverse().forEach((attempt) => {
      const button = element('button', 'attempt-row');
      button.type = 'button';
      button.classList.toggle('is-selected', attempt.id === selected?.id);
      button.setAttribute('aria-pressed', String(attempt.id === selected?.id));
      button.append(
        element('strong', '', `Attempt ${attempt.number}`),
        statusPill(attempt.status),
        element('span', 'attempt-row__duration', duration(attempt)),
      );
      button.addEventListener('click', () =>
        this.dispatch({ type: 'select-attempt', attemptId: attempt.id }),
      );
      list.append(button);
    });
    this.attempts.append(list);
  }

  private renderDetail(attempt?: StageAttempt): void {
    this.detail.replaceChildren();
    if (!attempt) {
      this.detail.append(element('p', 'empty', 'Select an attempt to inspect logs and artifacts.'));
      return;
    }
    this.detail.append(element('h2', '', `Attempt ${attempt.number}`));
    if (attempt.summary)
      this.detail.append(element('p', 'summary', redactSecrets(attempt.summary)));
    const tabs = element('div', 'detail-grid');
    const logs = element('section', 'logs');
    logs.append(element('h3', '', `Logs (${attempt.logs.length})`));
    const pre = element(
      'pre',
      '',
      attempt.logs.length
        ? attempt.logs.map(redactSecrets).join('\n')
        : 'No log output has been reported.',
    );
    pre.tabIndex = 0;
    logs.append(pre);
    const artifacts = element('section', 'artifacts');
    artifacts.append(element('h3', '', `Artifacts (${attempt.artifacts.length})`));
    if (attempt.artifacts.length) {
      attempt.artifacts.forEach((item) => artifacts.append(this.renderArtifact(item)));
    } else {
      artifacts.append(element('p', 'empty', 'No artifacts reported.'));
    }
    tabs.append(logs, artifacts);
    this.detail.append(tabs);
    requestAnimationFrame(() => {
      pre.scrollTop = pre.scrollHeight;
    });
  }

  private renderArtifact(artifact: Artifact): HTMLElement {
    const card = element('article', 'artifact');
    const header = element('div', 'artifact__header');
    header.append(element('strong', '', artifact.name), element('span', 'muted', artifact.kind));
    card.append(header);
    const url = safeArtifactUrl(artifact.url);
    if (artifact.preview) {
      card.append(element('pre', 'artifact__text', redactSecrets(artifact.preview)));
    } else if (url && artifact.kind === 'image') {
      const image = element('img');
      image.src = url;
      image.alt = `${artifact.name} preview`;
      image.loading = 'lazy';
      card.append(image);
    } else if (url && artifact.kind === 'video') {
      const video = element('video');
      video.src = url;
      video.controls = true;
      video.preload = 'metadata';
      card.append(video);
    } else if (url && artifact.kind === 'audio') {
      const audio = element('audio');
      audio.src = url;
      audio.controls = true;
      audio.preload = 'metadata';
      card.append(audio);
    }
    if (url) {
      const link = element('a', 'artifact__link', 'Open artifact');
      link.href = url;
      link.target = '_blank';
      link.rel = 'noopener';
      card.append(link);
    } else if (artifact.url) {
      card.append(element('span', 'muted', 'Preview unavailable for an external location'));
    }
    return card;
  }

  private renderActions(run?: PipelineRun, stage?: PipelineStage, attempt?: StageAttempt): void {
    this.actions.replaceChildren();
    if (!run) return;
    const commands: PipelineCommand[] = [];
    if (stage?.status === 'waiting_human') {
      if (attempt) commands.push('approve');
      commands.push('reject');
    }
    if (stage?.status === 'failed') commands.push('retry');
    if (run.status === 'running') commands.push('pause');
    if (run.status === 'paused') commands.push('resume');
    if (!['succeeded', 'failed', 'canceled'].includes(run.status)) commands.push('cancel');
    for (const command of [...new Set(commands)]) {
      const button = element(
        'button',
        `action action--${command}`,
        command[0].toUpperCase() + command.slice(1),
      );
      button.type = 'button';
      button.addEventListener(
        'click',
        () => void this.command(command, run, stage, attempt, button),
      );
      this.actions.append(button);
    }
  }

  private async command(
    command: PipelineCommand,
    run: PipelineRun,
    stage: PipelineStage | undefined,
    attempt: StageAttempt | undefined,
    button: HTMLButtonElement,
  ): Promise<void> {
    if (
      ['reject', 'cancel'].includes(command) &&
      !window.confirm(`${command === 'cancel' ? 'Cancel' : 'Reject'} ${run.name}?`)
    ) {
      return;
    }
    button.disabled = true;
    try {
      await sendCommand(
        run.id,
        command,
        stage?.id,
        attempt?.id,
        approvalArtifacts(attempt?.artifacts ?? []),
      );
      this.dispatch({ type: 'error', message: undefined });
    } catch (error) {
      this.dispatch({
        type: 'error',
        message: error instanceof Error ? error.message : 'Command failed',
      });
      button.disabled = false;
    }
  }

  private async submitMessage(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    const { run, stage } = selectedEntities(this.state);
    const message = this.input.value.trim();
    if (!run || !message) return;
    const button = this.form.querySelector<HTMLButtonElement>('button')!;
    button.disabled = true;
    try {
      await sendMessage(run.id, message, stage?.id);
      this.input.value = '';
      this.dispatch({ type: 'error', message: undefined });
    } catch (error) {
      this.dispatch({
        type: 'error',
        message: error instanceof Error ? error.message : 'Message failed',
      });
    } finally {
      button.disabled = false;
    }
  }
}

new Dashboard().start();

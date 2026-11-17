(() => {
  'use strict';

  /* ---------- Helpers ---------- */
  const KEY = 'ragchat.v1';
  const $ = (s, el = document) => el.querySelector(s);
  const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));



  if (!window.pdfjsLib) {
    console.error("PDF.js failed to load.");
  } else {
    pdfjsLib.GlobalWorkerOptions.workerSrc =
      "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

    console.log("PDF.js loaded successfully:", pdfjsLib.version);
  }



  marked.setOptions({
    gfm: true,
    breaks: false
  });




  let selectedAnswerId = null;

  function renderMarkdown(text) {
    let markdown = String(text || '');
    const mathExpressions = [];

    // Protect explicitly delimited math before Markdown parsing.
  markdown = markdown.replace(
      /\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$/g,
      expression => {
          const index = mathExpressions.push(expression) - 1;
          return `MATHPLACEHOLDER${index}TOKEN`;
      }
  );

    // Convert Markdown into HTML.
    let rawHtml = marked.parse(markdown);

    // Restore math expressions safely.
    rawHtml = rawHtml.replace(
      /MATHPLACEHOLDER(\d+)TOKEN/g,
      (match, index) => esc(mathExpressions[Number(index)])
    );

    return DOMPurify.sanitize(rawHtml);
  }





  const svg = (p, s = 18) => `<svg viewBox="0 0 24 24" width="${s}" height="${s}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${p}</svg>`;
  const icons = {
    dots: '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true"><circle cx="12" cy="5" r="1.8"/><circle cx="12" cy="12" r="1.8"/><circle cx="12" cy="19" r="1.8"/></svg>',
    folder: svg('<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'),
    chat: svg('<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/>'),
    plus: svg('<path d="M12 5v14M5 12h14"/>', 16),
    send: svg('<path d="M5 12h14M13 6l6 6-6 6"/>', 20),
    up: svg('<path d="M7 11v9H4v-9zM7 11l4-8a2 2 0 0 1 2 2v4h6a2 2 0 0 1 2 2.3l-1.2 7A2 2 0 0 1 17.8 20H7"/>', 15),
    down: svg('<path d="M17 13V4h3v9zM17 13l-4 8a2 2 0 0 1-2-2v-4H5a2 2 0 0 1-2-2.3l1.2-7A2 2 0 0 1 6.2 4H17"/>', 15),
    edit: svg('<path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>', 16),
    trash: svg('<path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14M10 11v5M14 11v5"/>', 16),
    chev: svg('<path d="M9 6l6 6-6 6"/>', 14),
    menu: svg('<path d="M4 6h16M4 12h16M4 18h16"/>', 20),
    bot: svg('<rect x="4" y="8" width="16" height="11" rx="3"/><path d="M12 8V4M9 13h.01M15 13h.01"/>', 18),
    spark: svg('<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z"/>', 18)
  };
  const FALLBACK = [
    'Based on the documents, the main idea is to retrieve relevant passages first and answer from them.',
    'This is a placeholder answer. A real model can be connected later.',
    'I found a few related passages that point to the same conclusion.'
  ];
  function getSourceLabel(source) {
    if (!source) return '';

    const value = String(source).toLowerCase();

    if (value.includes('hybrid')) return 'Hybrid';
    if (value.includes('web')) return 'Web Search';
    if (
        value.includes('document') ||
        value.includes('retriev') ||
        value.includes('rag')
    ) return 'Documents';

    if (
        value.includes('llm') ||
        value.includes('general')
    ) return 'LLM';

    return String(source);
  }

  /* ---------- State (persisted in localStorage) ---------- */
  const blank = () => ({ projects: [], chats: [], activeProjectId: null, activeChatId: null, expanded: [] });
  function load() {
    try {
      const d = JSON.parse(localStorage.getItem(KEY));
      if (d && Array.isArray(d.projects) && Array.isArray(d.chats)) return Object.assign(blank(), d);
    } catch (e) { /* ignore corrupt data */ }
    return blank();
  }
  let state = load();
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* storage full or blocked */ } };
  async function apiRequest(url, options = {}) {
  const response = await fetch(url, options);

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || error.error || `API Error: ${response.status}`);
  }

  return response.json();
}
  const typing = new Set();

  const chatsOf = pid => state.chats.filter(c => c.projectId === pid);
  const activeChat = () => state.chats.find(c => c.id === state.activeChatId) || null;
  const projectById = id => state.projects.find(p => p.id === id);



  // ---------- Project document viewer ----------
  const documentPanels = new Map();
  const selectedDocuments = new Map();
  const collapsedDocumentLists = new Set();

  const documentBaseUrl = (projectId, filename) =>
    `/api/projects/${encodeURIComponent(projectId)}/documents/${encodeURIComponent(filename)}`;




  
  // ---------- Retrieved document highlighting ----------

  function getDocumentChunks(projectId, filename) {
    const chat = activeChat();

    if (!chat || chat.projectId !== projectId || !selectedAnswerId) {
      return [];
    }

    const selectedMessage = chat.messages.find(
      message =>
        message.role === 'bot' &&
        message.id === selectedAnswerId
    );

    console.log(
        "FULL SELECTED MESSAGE:",
        JSON.stringify(selectedMessage, null, 2)
    );


    console.log("Selected message chunks:", selectedMessage?.retrievedChunks);

    console.log(
        "Chunk source metadata:",
        selectedMessage?.retrievedChunks?.map(chunk => chunk.metadata)
    );

    console.log("Currently selected filename:", filename);

    console.log("Selected answer ID:", selectedAnswerId);
    console.log("Retrieved chunks:", selectedMessage?.retrievedChunks);
    console.log("Selected filename:", filename);

    return (selectedMessage?.retrievedChunks || []).filter(chunk => {
      const metadata = chunk.metadata || {};
      const sourceName =
        metadata.filename ||
        String(metadata.source || '').split(/[\\/]/).pop();

      return (
        sourceName &&
        sourceName.toLowerCase() === filename.toLowerCase() &&
        chunk.content
      );
    });
  }

  function normalizeWithPositions(value) {
    let normalized = '';
    const positions = [];
    let previousWasSpace = false;

    for (let i = 0; i < value.length; i++) {
      const char = value[i];

      if (/\s/.test(char)) {
        if (!previousWasSpace && normalized.length) {
          normalized += ' ';
          positions.push(i);
        }
        previousWasSpace = true;
      } else {
        normalized += char.toLowerCase();
        positions.push(i);
        previousWasSpace = false;
      }
    }

    return { normalized, positions };
  }

  function highlightText(value, chunks = []) {

    console.log("Highlight function called");
    console.log("Chunks received:", chunks);

    const text = String(value ?? '');
    if (!text || !chunks.length) return esc(text);

    const source = normalizeWithPositions(text);
    const ranges = [];

    chunks.forEach((chunk, chunkIndex) => {
      const content = String(chunk.content || '').trim();
      if (!content) return;

      const candidates = [content];

      // If the complete chunk is not present in this paragraph/cell,
      // try meaningful sentences from that chunk.
      const sentences = content.match(/[^.!?]+[.!?]?/g) || [];
      candidates.push(
        ...sentences
          .map(sentence => sentence.trim())
          .filter(sentence => sentence.length >= 35)
      );

      const tried = new Set();

      for (const candidate of candidates) {
        const target = normalizeWithPositions(candidate).normalized;

        if (!target || tried.has(target)) continue;
        tried.add(target);

        let from = 0;
        let found = false;

        while (from < source.normalized.length) {
          const index = source.normalized.indexOf(target, from);
          if (index === -1) break;

          found = true;

          const originalStart = source.positions[index];
          const lastPosition = source.positions[index + target.length - 1];

          if (
            originalStart !== undefined &&
            lastPosition !== undefined
          ) {
            ranges.push({
              start: originalStart,
              end: lastPosition + 1,
              chunkIndex
            });
          }

          from = index + target.length;
        }

        if (found) break;
      }
    });

    if (!ranges.length) return esc(text);

    ranges.sort((a, b) => a.start - b.start);

    // Merge overlapping matches to avoid nested <mark> elements.
    const merged = [];

    for (const range of ranges) {
      const last = merged[merged.length - 1];

      if (last && range.start <= last.end) {
        last.end = Math.max(last.end, range.end);
      } else {
        merged.push({ ...range });
      }
    }

    let result = '';
    let cursor = 0;

    for (const range of merged) {
      result += esc(text.slice(cursor, range.start));
      result += `<mark class="doc-highlight" title="Retrieved passage">${esc(
        text.slice(range.start, range.end)
      )}</mark>`;
      cursor = range.end;
    }

    result += esc(text.slice(cursor));
    return result;
  }









  function documentTable(columns, rows, chunks = []) {
    if (!columns?.length) {
      return '<p class="doc-empty">This sheet has no data.</p>';
    }

    return `
      <div class="doc-table-wrap">
        <table class="doc-table">
          <thead>
            <tr>${columns.map(cell => `<th>${esc(cell ?? '')}</th>`).join('')}</tr>
          </thead>
          <tbody>
            ${rows.map(row => `
              <tr>
                ${columns.map((_, index) =>
                  `<td>${highlightText(row[index] ?? '', chunks)}</td>`
                ).join('')}
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  }

  // Configure PDF.js worker.
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';


  async function renderPDFPreview(chunks = []) {
    console.log("1. renderPDFPreview called");

    const viewer = document.getElementById('pdf-viewer');
    console.log("2. PDF viewer element:", viewer);

    if (!viewer) return;

    const pdfUrl = viewer.dataset.pdfUrl;
    console.log("3. PDF URL:", pdfUrl);

    CSS.highlights?.delete('rag-pdf-highlight');

    try {
      console.log("4. Starting PDF download...");

      const loadingTask = pdfjsLib.getDocument(pdfUrl);

      loadingTask.onProgress = (progress) => {
        console.log("PDF download progress:", progress);
      };

      const pdf = await loadingTask.promise;

      console.log("5. PDF loaded. Pages:", pdf.numPages);

      // Ignore this render if the viewer was replaced.
      if (!viewer.isConnected) return;

      viewer.innerHTML = '';

      for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber++) {
        const page = await pdf.getPage(pageNumber);

        if (!viewer.isConnected) return;

        const scale = 1.25;
        const viewport = page.getViewport({ scale });

        const pageContainer = document.createElement('div');
        pageContainer.className = 'pdf-page';
        pageContainer.style.width = `${viewport.width}px`;
        pageContainer.style.height = `${viewport.height}px`;

        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        canvas.width = viewport.width;
        canvas.height = viewport.height;

        const textLayer = document.createElement('div');
        textLayer.className = 'textLayer';

        pageContainer.appendChild(canvas);
        pageContainer.appendChild(textLayer);
        viewer.appendChild(pageContainer);

        await page.render({
          canvasContext: context,
          viewport
        }).promise;

        const textContent = await page.getTextContent();

        await pdfjsLib.renderTextLayer({
          textContent,
          container: textLayer,
          viewport,
          textDivs: []
        }).promise;

        highlightPDFPage(textLayer, chunks, pageNumber - 1);
      }

      console.log("6. PDF rendering completed.");

    } catch (error) {
      console.error('PDF rendering error:', error);

      viewer.innerHTML =
        '<div class="doc-empty">Could not render this PDF.</div>';
    }
  }

  function highlightPDFPage(textLayer, chunks = [], pageNumber = null) {
    if (!window.CSS?.highlights || typeof Highlight === 'undefined') {
      console.warn('CSS Custom Highlight API is not supported.');
      return;
    }

    function normalize(text) {
      return text
        .normalize('NFKC')
        .replace(/\u00ad/g, '')
        .toLowerCase()
        .replace(/\s+/g, ' ')
        .trim();
    }

    // Build searchable page text while retaining original DOM positions.
    const walker = document.createTreeWalker(
      textLayer,
      NodeFilter.SHOW_TEXT
    );

    const chars = [];
    const positions = [];

    while (walker.nextNode()) {
      const node = walker.currentNode;

      for (let i = 0; i < node.textContent.length; i++) {
        const normalizedChar = node.textContent[i]
          .normalize('NFKC')
          .toLowerCase();

        for (const char of normalizedChar) {
          const value = /\s/.test(char) ? ' ' : char;

          if (value === ' ' && chars[chars.length - 1] === ' ') {
            continue;
          }

          chars.push(value);
          positions.push({ node, offset: i });
        }
      }
    }

    const pageText = chars.join('');
    const ranges = [];

    for (const chunk of chunks) {
      const metadata = chunk.metadata || {};

      // Backend page metadata is zero-based.
      if (
        pageNumber !== null &&
        metadata.page !== undefined &&
        Number(metadata.page) !== pageNumber
      ) {
        continue;
      }

      // Remove the artificial section-context prefix.
      const content = chunk.content.replace(
        /^Section context:.*(?:\r?\n)+/i,
        ''
      );

      const words = normalize(content).split(' ').filter(Boolean);

      // Search smaller overlapping phrases instead of the whole chunk.
      const windowSize = 10;
      const step = 6;

      for (let i = 0; i < words.length; i += step) {
        const phrase = words.slice(i, i + windowSize).join(' ');

        if (phrase.split(' ').length < 5) continue;

        let start = pageText.indexOf(phrase);

        while (start !== -1) {
          const end = start + phrase.length - 1;

          if (positions[start] && positions[end]) {
            const range = document.createRange();

            range.setStart(
              positions[start].node,
              positions[start].offset
            );

            range.setEnd(
              positions[end].node,
              positions[end].offset + 1
            );

            ranges.push(range);
          }

          start = pageText.indexOf(phrase, start + 1);
        }
      }
    }

    console.log(
      `PDF page ${pageNumber + 1}: ${ranges.length} highlight matches`
    );

    if (ranges.length) {
      const existing = CSS.highlights.get('rag-pdf-highlight');

      CSS.highlights.set(
        'rag-pdf-highlight',
        new Highlight(...(existing ? [...existing, ...ranges] : ranges))
      );
    }
  }


  function documentPreviewMarkup(projectId, filename, preview, chunks = []) {
    if (!preview) {
      return '<div class="doc-empty">Select a document to preview it.</div>';
    }

    const url = documentBaseUrl(projectId, filename);

    if (preview.type === 'pdf') {
      return `
        <div
          class="doc-pdf"
          id="pdf-viewer"
          data-pdf-url="${url}/preview"
        >
          <div class="doc-empty">Loading PDF...</div>
        </div>
      `;
    }

    if (preview.type === 'txt') {
      return `<pre class="doc-text">${highlightText(preview.text, chunks)}</pre>`;
    }

    if (preview.type === 'docx') {
      const paragraphs = (preview.paragraphs || [])
        .map(text => `<p>${highlightText(text, chunks)}</p>`)
        .join('');

      const tables = (preview.tables || [])
        .map(table => {
          if (!table.length) return '';
          return documentTable(
            table[0],
            table.slice(1),
            chunks
          );
        })
        .join('');

      return `<div class="doc-document">${paragraphs}${tables}</div>`;
    }

    if (preview.type === 'table') {
      return `
        ${documentTable(preview.columns, preview.rows, chunks)}
        ${preview.truncated ? '<p class="doc-note">Preview limited to 200 rows.</p>' : ''}
      `;
    }

    if (preview.type === 'xlsx') {
      return (preview.sheets || []).map(sheet => `
        <section class="doc-sheet">
          <h4>${esc(sheet.name)}</h4>
          ${documentTable(sheet.columns, sheet.rows, chunks)}
          ${sheet.truncated ? '<p class="doc-note">Preview limited to 200 rows.</p>' : ''}
        </section>
      `).join('');
    }

    return '<div class="doc-empty">Preview is not available for this file.</div>';
  }


  function renderDocumentPanel(projectId) {
    const panel = $('#document-panel');
    if (!panel) return;

    const data = documentPanels.get(projectId) || {
      loading: true,
      documents: [],
      preview: null
    };

    const selected = selectedDocuments.get(projectId);
    const collapsed = collapsedDocumentLists.has(projectId);

    const retrievedChunks = selected ? getDocumentChunks(projectId, selected): [];


    panel.classList.toggle('documents-collapsed', collapsed);

    const documentItems = data.documents.map(doc => `
      <button
        class="doc-file ${doc.name === selected ? 'active' : ''}"
        data-action="select-document"
        data-project-id="${esc(projectId)}"
        data-filename="${esc(doc.name)}"
        title="${esc(doc.name)}"
      >
        <span class="doc-file-icon">${esc(doc.extension.replace('.', '').toUpperCase())}</span>
        <span class="doc-file-name">${esc(doc.name)}</span>
      </button>
    `).join('');

    let previewContent = '';

    if (data.loadingPreview) {
      previewContent = '<div class="doc-empty">Loading preview...</div>';
    } else if (data.error) {
      previewContent = `<div class="doc-empty">${esc(data.error)}</div>`;
    } else if (selected && data.preview) {
      previewContent = documentPreviewMarkup(
        projectId,
        selected,
        data.preview,
        retrievedChunks
      );
    } else if (data.loading) {
      previewContent = '<div class="doc-empty">Loading documents...</div>';
    } else {
      previewContent = '<div class="doc-empty">No documents available.</div>';
    }

    const downloadUrl = selected
      ? `${documentBaseUrl(projectId, selected)}/download`
      : '';

    panel.innerHTML = `
      <div class="doc-panel-header">
        <div class="doc-panel-title">
          <strong>Documents</strong>
          <span>${data.documents.length} file${data.documents.length === 1 ? '' : 's'}</span>
        </div>

        <button
          class="doc-collapse"
          data-action="toggle-documents"
          data-project-id="${esc(projectId)}"
          aria-label="${collapsed ? 'Show document list' : 'Hide document list'}"
          title="${collapsed ? 'Show document list' : 'Hide document list'}"
        >
          ${icons.chev}
        </button>
      </div>

      <div class="doc-list">
        ${data.loading
          ? '<p class="doc-empty">Loading documents...</p>'
          : documentItems || '<p class="doc-empty">No documents found.</p>'
        }
      </div>

      <div class="doc-preview-header">
        <span title="${esc(selected || '')}">${esc(selected || 'Document preview')}</span>
        ${selected ? `
          <a class="doc-download" href="${downloadUrl}" download>
            Download
          </a>
        ` : ''}
      </div>

      <div class="doc-preview">
        ${previewContent}
      </div>
    `;

    // Render PDF after the viewer element has been added to the DOM.
    if (selected && data.preview?.type === 'pdf') {
      console.log("Calling renderPDFPreview...");
      renderPDFPreview(retrievedChunks);
    }


  }

  async function loadDocumentPreview(projectId, filename) {
    const data = documentPanels.get(projectId);
    if (!data || !filename) return;

    data.error = null;
    data.loadingPreview = true;
    data.preview = null;
    renderDocumentPanel(projectId);

    // PDFs are displayed directly by the browser.
    if (filename.toLowerCase().endsWith('.pdf')) {
      data.preview = { type: 'pdf' };
      data.loadingPreview = false;
      renderDocumentPanel(projectId);
      return;
    }

    try {
      const preview = await apiRequest(
        `${documentBaseUrl(projectId, filename)}/preview`
      );

      // Ignore an older request if the user selected another file.
      if (selectedDocuments.get(projectId) !== filename) return;

      data.preview = preview;
    } catch (error) {
      if (selectedDocuments.get(projectId) !== filename) return;
      data.error = error.message || 'Could not load document preview.';
    } finally {
      if (selectedDocuments.get(projectId) === filename) {
        data.loadingPreview = false;
        renderDocumentPanel(projectId);
      }
    }
  }

  async function loadProjectDocuments(projectId) {
    let data = documentPanels.get(projectId);

    if (data?.loaded || data?.loading) {
      renderDocumentPanel(projectId);
      return;
    }

    data = {
      loading: true,
      loaded: false,
      documents: [],
      preview: null,
      loadingPreview: false,
      error: null
    };

    documentPanels.set(projectId, data);
    renderDocumentPanel(projectId);

    try {
      const result = await apiRequest(
        `/api/projects/${encodeURIComponent(projectId)}/documents`
      );

      data.documents = result.documents || [];
      data.loaded = true;

      if (
        !selectedDocuments.has(projectId) ||
        !data.documents.some(
          doc => doc.name === selectedDocuments.get(projectId)
        )
      ) {
        selectedDocuments.set(
          projectId,
          data.documents[0]?.name || null
        );
      }

      data.loading = false;
      renderDocumentPanel(projectId);

      const selected = selectedDocuments.get(projectId);
      if (selected) {
        await loadDocumentPreview(projectId, selected);
      }
    } catch (error) {
      data.loading = false;
      data.error = error.message || 'Could not load project documents.';
      renderDocumentPanel(projectId);
    }
  }

  function selectDocument(projectId, filename) {
    selectedDocuments.set(projectId, filename);

    const data = documentPanels.get(projectId);
    if (data) {
      data.preview = null;
      data.error = null;
    }

    renderDocumentPanel(projectId);
    loadDocumentPreview(projectId, filename);
  }

  function attachPanelResizer() {
    const workspace = $('.workspace');
    const resizer = $('#panel-resizer');
    const panel = $('#document-panel');

    if (!workspace || !resizer || !panel) return;

    resizer.addEventListener('pointerdown', event => {
      if (window.innerWidth <= 800) return;

      event.preventDefault();

      const startX = event.clientX;
      const startWidth = panel.getBoundingClientRect().width;
      const workspaceWidth = workspace.getBoundingClientRect().width;

      const onMove = moveEvent => {
        const requestedWidth =
          startWidth + (startX - moveEvent.clientX);

        const maxWidth = workspaceWidth * 0.6;
        const width = Math.max(
          260,
          Math.min(requestedWidth, maxWidth)
        );

        workspace.style.setProperty(
          '--document-panel-width',
          `${width}px`
        );
      };

      const onUp = () => {
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
      };

      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
    });
  }






  /* ---------- Sidebar ---------- */
  function chatRow(c) {
    return `<div class="item ${c.id === state.activeChatId ? 'active' : ''}" tabindex="0" data-action="open-chat" data-id="${c.id}">
      <span class="item-label">${icons.chat}<span class="t">${esc(c.title)}</span></span>
      <button class="kebab" data-menu="chat" data-id="${c.id}" aria-label="Chat options">${icons.dots}</button>
    </div>`;
  }

  function renderSidebar() {
    const projects = state.projects.map(p => {
      const open = state.expanded.includes(p.id);
      const kids = chatsOf(p.id);
      return `<div>
        <div class="item ${p.id === state.activeProjectId && !state.activeChatId ? 'active' : ''}" tabindex="0" data-action="open-project" data-id="${p.id}">
          <span class="item-label"><span class="chev ${open ? 'open' : ''}">${icons.chev}</span>${icons.folder}<span class="t">${esc(p.name)}</span></span>
          <button class="kebab" data-menu="project" data-id="${p.id}" aria-label="Project options">${icons.dots}</button>
        </div>
        ${open ? `<div class="nested">${kids.map(chatRow).join('')}<button class="link-btn" data-action="new-project-chat" data-id="${p.id}">${icons.plus}New chat</button></div>` : ''}
      </div>`;
    }).join('') || '<p class="hint">No projects yet. Create one to group related chats.</p>';

    const general = chatsOf(null).map(chatRow).join('') || '<p class="hint">No chats yet. Start a new conversation.</p>';

    $('#sidebar').innerHTML = `
      <div class="brand"><span class="brand-mark">${icons.spark}</span>RAG Chatbot</div>
      <div class="side-scroll">
        <div class="section-head">Projects</div>
        <button class="btn-new" data-action="new-project">${icons.plus}Create project</button>
        ${projects}
        <div class="section-head">General chatbot</div>
        <button class="btn-new" data-action="new-general-chat">${icons.plus}New chat</button>
        ${general}
      </div>`;
  }

  /* ---------- Main area ---------- */
  const topbar = (ctx, title) => `<header class="topbar">
    <button class="icon-btn" data-action="toggle-sidebar" aria-label="Open sidebar">${icons.menu}</button>
    <div class="crumb"><div class="crumb-ctx">${esc(ctx)}</div><h1>${esc(title)}</h1></div></header>`;


  function renderMain() {
    const chat = activeChat();
    const proj = projectById(state.activeProjectId);
    const main = $('#main');

    if (chat) {
      const p = projectById(chat.projectId);

      const chatContent = `
        <div class="messages" id="messages">
          <div class="thread" id="thread"></div>
        </div>

        <div class="composer">
          <div class="composer-box">
            <textarea id="q" rows="1" placeholder="Ask a question…" aria-label="Your question"></textarea>

            <div class="token-tracker" id="token-tracker">
              <button class="token-trigger" data-action="token-usage"
                      aria-label="Token usage" title="Token usage">

                <svg class="token-ring" viewBox="0 0 36 36"
                    xmlns="http://www.w3.org/2000/svg">
                  <circle class="token-ring-bg" cx="18" cy="18" r="15"></circle>
                  <circle class="token-ring-progress" cx="18" cy="18" r="15"></circle>
                </svg>

                <span class="token-count">0</span>
              </button>

              <div class="token-popup" id="token-popup" hidden>
                <h4>Token usage</h4>
                <div class="token-stat">
                  <span>Input tokens</span>
                  <strong id="token-input">0</strong>
                </div>
                <div class="token-stat">
                  <span>Output tokens</span>
                  <strong id="token-output">0</strong>
                </div>
                <div class="token-stat">
                  <span>Total tokens</span>
                  <strong id="token-total">0</strong>
                </div>
                <div class="token-stat">
                  <span>Context usage</span>
                  <strong id="token-percentage">0%</strong>
                </div>
              </div>
            </div>

            <button class="send" data-action="send" aria-label="Send question">${icons.send}</button>
          </div>
        </div>
      `;

      main.innerHTML = topbar(
        p ? 'Project: ' + p.name : 'General chatbot',
        chat.title
      ) + `
        <div class="workspace">
          <section class="chat-panel">
            ${chatContent}
          </section>

          ${p ? `
            <div class="panel-resizer" id="panel-resizer" role="separator" aria-orientation="vertical"></div>
            <aside class="document-panel" id="document-panel"></aside>
          ` : ''}
        </div>
      `;

      updateMessages(true);

      const q = $('#q');
      q.addEventListener('input', () => {
        q.style.height = 'auto';
        q.style.height = Math.min(q.scrollHeight, 160) + 'px';
      });

      q.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          send();
        }
      });

      if (p) {
        attachPanelResizer();
        loadProjectDocuments(p.id);
      }

      q.focus();

    } else if (proj) {
      const kids = chatsOf(proj.id);

      main.innerHTML = topbar('Project', proj.name) + `
        <div class="workspace">
          <section class="chat-panel">
            <div class="messages">
              <div class="proj-view">
                <div class="row">
                  <h2>Chats in this project</h2>
                  <button class="btn primary" data-action="new-project-chat" data-id="${proj.id}">
                    ${icons.plus}New chat
                  </button>
                </div>

                ${kids.length
                  ? kids.map(c => `
                    <div class="chat-card" tabindex="0" data-action="open-chat" data-id="${c.id}">
                      <div>
                        <div class="title">${esc(c.title)}</div>
                        <div class="sub">${c.messageCount ?? c.messages.length} message${(c.messageCount ?? c.messages.length) === 1 ? '' : 's'}</div>
                      </div>
                      <button class="kebab" data-menu="chat" data-id="${c.id}" aria-label="Chat options">
                        ${icons.dots}
                      </button>
                    </div>
                  `).join('')
                  : `
                    <div class="empty">
                      <div class="badge">${icons.chat}</div>
                      <h2>No chats yet</h2>
                      <p>Start a chat to ask questions in this project.</p>
                    </div>
                  `
                }
              </div>
            </div>
          </section>

          <div class="panel-resizer" id="panel-resizer" role="separator" aria-orientation="vertical"></div>
          <aside class="document-panel" id="document-panel"></aside>
        </div>
      `;

      attachPanelResizer();
      loadProjectDocuments(proj.id);

    } else {
      main.innerHTML = topbar('Welcome', 'RAG Chatbot') + `
        <div class="empty">
          <div class="badge">${icons.spark}</div>
          <h2>What would you like to ask?</h2>
          <p>Create a project to keep related chats together, or start a general chat.</p>
          <div class="empty-actions">
            <button class="btn primary" data-action="new-general-chat">${icons.plus}New chat</button>
            <button class="btn" data-action="new-project">${icons.plus}Create project</button>
          </div>
        </div>
      `;
    }
  }


  const CONTEXT_WINDOW_LIMIT = 131072;
  function updateTokenTracker(chat) {
    const countElement = document.querySelector('.token-count');
    const inputElement = document.getElementById('token-input');
    const outputElement = document.getElementById('token-output');
    const totalElement = document.getElementById('token-total');

    if (!chat) return;

    const usage = chat.messages.reduce((total, message) => {
      const u = message.usage || {};

      total.input += Number(u.input_tokens || 0);
      total.output += Number(u.output_tokens || 0);
      total.total += Number(u.total_tokens || 0);

      return total;
    }, { input: 0, output: 0, total: 0 });

    // Get the latest request's context usage
    const latestContextMessage = [...chat.messages]
      .reverse()
      .find(message => message.contextUsage);

    const contextTokens = Number(
      latestContextMessage?.contextUsage?.input_tokens || 0
    );

    const contextPercentage = Math.min(
      (contextTokens / CONTEXT_WINDOW_LIMIT) * 100,
      100
    );


    // console.log("Latest context message:", latestContextMessage);
    // console.log("Context tokens:", contextTokens);
    // console.log("Context limit:", CONTEXT_WINDOW_LIMIT);
    // console.log(
    //   "Calculated percentage:",
    //   (contextTokens / CONTEXT_WINDOW_LIMIT) * 100
    // );


    // Update circular indicator
    const progressCircle = document.querySelector('.token-ring-progress');

    if (progressCircle) {
      const circumference = 94.25;

      progressCircle.style.strokeDashoffset =
        circumference - (contextPercentage / 100) * circumference;
    }

    const percentageElement = document.getElementById('token-percentage');

    if (percentageElement) {
      percentageElement.textContent =
        `${contextPercentage.toFixed(2)}%`;
    }
    const format = value => value.toLocaleString('en-US');

    if (countElement) {
      countElement.textContent = usage.total >= 1000
        ? `${(usage.total / 1000).toFixed(1)}k`
        : String(usage.total);
    }

    if (inputElement) inputElement.textContent = format(usage.input);
    if (outputElement) outputElement.textContent = format(usage.output);
    if (totalElement) totalElement.textContent = format(usage.total);
  }



  function formatAssistantMessages() {
    document.querySelectorAll('.msg.bot .message-text').forEach(element => {

      // Render mathematical expressions.
      if (window.renderMathInElement) {
        renderMathInElement(element, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '\\[', right: '\\]', display: true },
            { left: '\\(', right: '\\)', display: false },
            { left: '$', right: '$', display: false }
          ],
          throwOnError: false
        });
      }

      // Highlight code blocks.
      if (window.hljs) {
        element.querySelectorAll('pre code').forEach(block => {
          if (!block.dataset.highlighted) {
            hljs.highlightElement(block);
          }
        });
      }
    });
  }




  function updateMessages(scroll) {
    const chat = activeChat();
    const thread = $('#thread');

    if (!chat || !thread) return;

    updateTokenTracker(chat);

    const botIcon = `<div class="avatar">${icons.bot}</div>`;


    let html = chat.messages.length
      ? ''
      : '<div class="empty-chat">Ask your first question below.</div>';

    html += chat.messages.map(m => m.role === 'user'
      ? `<div class="msg user">
          <div class="bubble">${esc(m.text)}</div>
        </div>`
      : `<div class="msg bot ${selectedAnswerId === m.id ? 'selected-answer' : ''}"
              data-action="select-answer"
              data-msg="${m.id}"
              style="cursor: pointer;">

          ${botIcon}

          <div class="bubble">
            <div class="message-text">${renderMarkdown(m.text)}</div>

            <div class="fb">
              ${m.answerSource ? `
                <div class="answer-source">
                  <span class="source-dot"></span>
                  <span>${esc(getSourceLabel(m.answerSource))}</span>
                </div>
              ` : ''}

              <button
                class="up ${m.feedback === 'up' ? 'on' : ''}"
                data-action="fb"
                data-msg="${m.id}"
                data-v="up"
                aria-pressed="${m.feedback === 'up'}"
                aria-label="Good answer">
                ${icons.up}
              </button>

              <button
                class="down ${m.feedback === 'down' ? 'on' : ''}"
                data-action="fb"
                data-msg="${m.id}"
                data-v="down"
                aria-pressed="${m.feedback === 'down'}"
                aria-label="Bad answer">
                ${icons.down}
              </button>
            </div>
          </div>
        </div>`
    ).join('');

    if (typing.has(chat.id)) {
      html += `
        <div class="msg bot">
          ${botIcon}
          <div class="bubble typing">
            <i></i><i></i><i></i>
          </div>
        </div>`;
    }

    thread.innerHTML = html;
    formatAssistantMessages();

    if (scroll) {
      $('#messages').scrollTop = 1e9;
    }
  }

  function render() { renderSidebar(); renderMain(); }
  async function initialize() {
  try {
    const [projectsData, chatsData] = await Promise.all([
      apiRequest('/api/projects'),
      apiRequest('/api/chats')
    ]);

    // Convert backend projects into frontend format
    state.projects = Object.entries(projectsData).map(([id, project]) => ({
      id,
      name: project.name
    }));

    // Convert backend chats into frontend format
    state.chats = Object.entries(chatsData).map(([id, chat]) => ({
        id,
        projectId: chat.mode === 'project' ? chat.rag_id : null,
        ragId: chat.rag_id,
        mode: chat.mode,
        title: chat.title,
        titled: chat.title !== 'New Chat',
        messageCount: chat.message_count || 0,
        messages: []
    }));

    // Keep only valid selections
    if (!state.projects.some(p => p.id === state.activeProjectId)) {
      state.activeProjectId = null;
    }

    if (!state.chats.some(c => c.id === state.activeChatId)) {
      state.activeChatId = null;
    }

    state.expanded = state.expanded.filter(id =>
      state.projects.some(p => p.id === id)
    );

    save();
    render();

    // Restore the previously selected chat's conversation
    const activeChat = state.chats.find(
        c => c.id === state.activeChatId
    );

    if (activeChat) {
        await selectChat(activeChat);
    }

  } catch (error) {
    console.error('Backend loading failed:', error);

    $('#main').innerHTML = `
      <div class="empty">
        <h2>Backend connection failed</h2>
        <p>${esc(error.message)}</p>
        <button class="btn primary" onclick="location.reload()">Retry</button>
      </div>
    `;
  }
}

  /* ---------- Chat logic ---------- */
  async function getAnswer() {
    try {
      const r = await fetch('/api/answer');
      if (!r.ok) throw new Error('bad status');
      return (await r.json()).answer;
    } catch (e) {
      return FALLBACK[Math.floor(Math.random() * FALLBACK.length)];
    }
  }


  async function send() {
      const q = $('#q');
      const chat = activeChat();

      if (!q || !chat || typing.has(chat.id)) return;

      const text = q.value.trim();
      if (!text) return;

      const chatId = chat.id;

      // 1. Add the user's message
      chat.messages.push({
          id: uid(),
          role: 'user',
          text
      });

      // 2. Set the chat title using the first question
      if (!chat.titled) {
          chat.title = text.length > 40
              ? text.slice(0, 40) + '…'
              : text;

          chat.titled = true;

          // Save the title in the Python backend
          try {
              await apiRequest(`/api/chats/${chatId}`, {
                  method: 'PATCH',
                  headers: {
                      'Content-Type': 'application/json'
                  },
                  body: JSON.stringify({
                      title: chat.title
                  })
              });
          } catch (error) {
              console.error('Chat title update failed:', error);
          }
      }

      save();

      q.value = '';
      q.style.height = 'auto';

      typing.add(chatId);

      renderSidebar();
      updateMessages(true);

      // Assistant message will be created when the first token arrives
      let botMessage = null;
      let buffer = '';
      let streamError = null;
      let receivedDone = false;

      try {
          // 3. Send the question to FastAPI
          const response = await fetch('/api/chat/stream', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json'
              },
              body: JSON.stringify({
                  chat_id: chatId,
                  query: text
              })
          });

          if (!response.ok) {
              const error = await response.json().catch(() => ({}));
              throw new Error(
                  error.detail || error.error || `API Error: ${response.status}`
              );
          }

          if (!response.body) {
              throw new Error('Streaming response is not available.');
          }

          // 4. Read the streaming response
          const reader = response.body.getReader();
          const decoder = new TextDecoder();

          function processEvent(block) {
              const dataLine = block
                  .split(/\r?\n/)
                  .find(line => line.startsWith('data:'));

              if (!dataLine) return;

              // const event = JSON.parse(dataLine.slice(5).trim());
              const event = JSON.parse(dataLine.slice(5).trim());

              if (event.type === 'token') {
                  console.log('TOKEN:', JSON.stringify(event.content));
              }

              if (event.type === 'done') {
                  console.log('FINAL DONE EVENT:', event);
                  console.log('FINAL FRONTEND TEXT:', botMessage?.text);
              }
              // console.log('SSE event received:', event);

              // 5. Receive answer tokens
              if (event.type === 'token') {
                  if (!botMessage) {
                      botMessage = {
                          id: uid(),
                          role: 'bot',
                          text: '',
                          feedback: null
                      };

                      chat.messages.push(botMessage);
                      typing.delete(chatId);
                  }

                  botMessage.text += event.content || '';

                  save();

                  if (state.activeChatId === chatId) {
                      updateMessages(true);
                  }
              }

              // 6. Receive final response metadata
              else if (event.type === 'done') {
                  receivedDone = true;

                  if (!botMessage) {
                      botMessage = {
                          id: uid(),
                          role: 'bot',
                          text: '',
                          feedback: null
                      };

                      chat.messages.push(botMessage);
                  }

                  botMessage.responseId = event.response_id;
                  botMessage.answerSource = event.answer_source;
                  botMessage.feedbackQuery = event.feedback_query;
                  botMessage.feedbackContext = event.feedback_context;
                  botMessage.usage = event.usage;
                  botMessage.contextUsage = event.context_usage;
                  botMessage.retrievedChunks = event.retrieved_chunks || [];
                  if (chat.projectId) {renderDocumentPanel(chat.projectId);}
                  // console.log("Bot message:", botMessage);
                  // console.log("Chat's latest message:", chat.messages.at(-1));
                  // console.log('Context usage received:', event.context_usage);
                  updateTokenTracker(chat);
              }

              // 7. Handle backend errors
              else if (event.type === 'error') {
                  streamError = event.message || 'Something went wrong.';
              }
          }

          while (true) {
              const { value, done } = await reader.read();

              if (done) break;

              buffer += decoder.decode(value, { stream: true });

              const blocks = buffer.split(/\r?\n\r?\n/);
              buffer = blocks.pop();

              for (const block of blocks) {
                  processEvent(block);

                  if (streamError) {
                      throw new Error(streamError);
                  }
              }
          }

          // Process any remaining event
          buffer += decoder.decode();

          if (buffer.trim()) {
              processEvent(buffer);
          }

          if (streamError) {
              throw new Error(streamError);
          }

          if (!receivedDone) {
              throw new Error('The response stream ended unexpectedly.');
          }

      } catch (error) {
          console.error('Message generation failed:', error);

          if (!botMessage) {
              botMessage = {
                  id: uid(),
                  role: 'bot',
                  text: '',
                  feedback: null
              };

              chat.messages.push(botMessage);
          }

          botMessage.text += botMessage.text
              ? `\n\nError: ${error.message}`
              : `Sorry, something went wrong: ${error.message}`;

      } finally {
          // 8. Finish the response
          typing.delete(chatId);

          save();

          if (state.activeChatId === chatId) {
              updateMessages(true);
              updateTokenTracker(chat);
          }
      }
  }


  /* ---------- CRUD ---------- */
  async function selectChat(c) {
      selectedAnswerId = null;
      state.activeChatId = c.id;
      state.activeProjectId = c.projectId;

      if (c.projectId && !state.expanded.includes(c.projectId)) {
          state.expanded.push(c.projectId);
      }
      

      // Show the selected chat immediately
      save();
      closeSidebar();
      render();

      try {
          const data = await apiRequest(
              `/api/chats/${c.id}/messages`
          );

          // Prevent an older request from overwriting a newly selected chat
          if (state.activeChatId !== c.id) return;

          c.messages = data.messages.map(message => ({
              id: message.id || uid(),
              role: message.role === 'assistant' ? 'bot' : message.role,
              text: message.text,
              usage: message.usage || null,
              answerSource: message.answer_source,
              feedbackQuery: message.feedback_query,
              feedbackContext: message.feedback_context,
              retrievedChunks: message.retrieved_chunks || [],
              responseId: message.response_id,
              feedback: message.feedback
          }));

          save();
          updateMessages(true);
          if (c.projectId) {renderDocumentPanel(c.projectId);}
      } catch (error) {
          console.error('Chat history loading failed:', error);
      }
  }



  async function newChat(projectId) {
    try {
      const data = await apiRequest('/api/chats', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          mode: projectId ? 'project' : 'general',
          rag_id: projectId || null
        })
      });

      const c = {
        id: data.chat_id,
        projectId: data.mode === 'project' ? data.rag_id : null,
        ragId: data.rag_id,
        mode: data.mode,
        title: data.title,
        titled: false,
        messages: []
      };

      state.chats.unshift(c);
      selectChat(c);

    } catch (error) {
      console.error('Chat creation failed:', error);
      alert(error.message || 'Could not create chat.');
    }
  }



  function openProject(id) {
    const open = state.expanded.includes(id);
    if (open && state.activeProjectId === id && !state.activeChatId) {
      state.expanded = state.expanded.filter(x => x !== id);
    } else if (!open) {
      state.expanded.push(id);
    }
    state.activeProjectId = id; state.activeChatId = null;
    save(); closeSidebar(); render();
  }



  function createProject() {
    modal({
      title: 'Create project',
      input: { label: 'Project name', value: '' },
      confirmText: 'Create',
      closeOnConfirm: false,

      extraHtml: `
        <label for="project-files">Upload documents</label>
        <input
          type="file"
          id="project-files"
          multiple
          accept=".pdf,.txt,.docx,.csv,.xlsx"
        />

        <div id="upload-progress" hidden style="margin-top:18px;">
          <div style="display:flex;justify-content:space-between;gap:10px;margin-bottom:8px;">
            <span id="upload-stage">Preparing documents...</span>
            <span id="upload-percent">0%</span>
          </div>

          <div style="height:8px;background:#e5e7eb;border-radius:10px;overflow:hidden;">
            <div
              id="upload-bar"
              style="height:100%;width:0%;background:#16a34a;transition:width .3s;"
            ></div>
          </div>

          <p id="upload-error" style="color:#dc2626;margin-top:8px;"></p>
        </div>
      `,

      onConfirm: async (name, { close, setProgress }) => {

        const fileInput = document.querySelector('#project-files');
        const files = fileInput?.files;

        if (!files || files.length === 0) {
          throw new Error('Please select at least one document.');
        }

        const formData = new FormData();

        formData.append('name', name);

        for (const file of files) {
          formData.append('files', file);
        }

        setProgress('Uploading documents...', 5);

        const response = await fetch('/api/projects/stream', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          const error = await response.json().catch(() => ({}));
          throw new Error(
            error.detail || error.error || 'Project creation failed.'
          );
        }

        if (!response.body) {
          throw new Error('Streaming response is not supported.');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        let buffer = '';
        let completedProject = null;

        function processEvents() {
          const blocks = buffer.split('\n\n');
          buffer = blocks.pop();

          for (const block of blocks) {
            const line = block
              .split('\n')
              .find(line => line.startsWith('data:'));

            if (!line) continue;

            const event = JSON.parse(line.slice(5).trim());

            if (event.type === 'error') {
              throw new Error(event.message);
            }

            if (event.type === 'done') {
              completedProject = event;
            } else if (event.stage) {
              setProgress(event.stage, event.progress);
            }
          }
        }

        while (true) {
          const { value, done } = await reader.read();

          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          processEvents();
        }

        buffer += decoder.decode();
        if (buffer.trim()) {
          buffer += '\n\n';
          processEvents();
        }

        if (!completedProject) {
          throw new Error('Project creation did not complete successfully.');
        }

        setProgress('Project created successfully!', 100);

        // Close the modal after successful creation
        close(true);

        // Update frontend state
        state.projects.unshift({
          id: completedProject.rag_id,
          name: completedProject.name
        });

        state.expanded.push(completedProject.rag_id);
        state.activeProjectId = completedProject.rag_id;
        state.activeChatId = null;

        save();
        render();
      }
    });
  }


  function renameProject(id) {
    const p = projectById(id); if (!p) return;
    modal({ title: 'Edit project', input: { label: 'Project name', value: p.name }, confirmText: 'Save', onConfirm: name => { p.name = name; save(); render(); } });
  }
  async function deleteProject(id) {
    const p = projectById(id);
    if (!p) return;

    const n = chatsOf(id).length;

    modal({
      title: 'Delete project?',
      body: `“${esc(p.name)}” and its ${n} chat${n === 1 ? '' : 's'} will be permanently deleted.`,
      confirmText: 'Delete project',
      danger: true,

      onConfirm: async () => {
        // Delete from backend first
        await apiRequest(`/api/projects/${encodeURIComponent(id)}`, {
          method: 'DELETE'
        });

        // Update frontend only after backend confirms success
        state.chats = state.chats.filter(c => c.projectId !== id);
        state.projects = state.projects.filter(x => x.id !== id);
        state.expanded = state.expanded.filter(x => x !== id);

        if (state.activeProjectId === id) {
          state.activeProjectId = null;
          state.activeChatId = null;
        }

        save();
        render();
      }
    });
  }
  function renameChat(id) {
    const c = state.chats.find(x => x.id === id); if (!c) return;
    modal({ title: 'Edit chat', input: { label: 'Chat name', value: c.title }, confirmText: 'Save', onConfirm: name => { c.title = name; c.titled = true; save(); render(); } });
  }

  async function deleteChat(id) {
    const c = state.chats.find(x => x.id === id);
    if (!c) return;

    modal({
      title: 'Delete chat?',
      body: `“${esc(c.title)}” will be permanently deleted.`,
      confirmText: 'Delete chat',
      danger: true,
      closeOnConfirm: false,

      onConfirm: async (_name, { close }) => {
        // Delete from backend first
        await apiRequest(`/api/chats/${encodeURIComponent(id)}`, {
          method: 'DELETE'
        });

        // Update frontend only after backend confirms success
        state.chats = state.chats.filter(x => x.id !== id);

        if (state.activeChatId === id) {
          state.activeChatId = null;
        }

        save();
        render();
        close(true);
      }
    });
  }

  /* ---------- Menu & modal ---------- */
  function closeMenu() { const m = $('#menu'); if (m) m.remove(); }
  function openMenu(btn, items) {
    closeMenu();
    const m = document.createElement('div');
    m.className = 'menu'; m.id = 'menu'; m.setAttribute('role', 'menu');
    m.innerHTML = items.map((it, i) => `<button role="menuitem" data-i="${i}" class="${it.danger ? 'danger' : ''}">${it.icon}${it.label}</button>`).join('');
    document.body.appendChild(m);
    const r = btn.getBoundingClientRect();
    m.style.top = Math.min(r.bottom + 4, window.innerHeight - m.offsetHeight - 8) + 'px';
    m.style.left = Math.max(8, Math.min(r.right - m.offsetWidth, window.innerWidth - m.offsetWidth - 8)) + 'px';
    m.addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      const it = items[+b.dataset.i]; closeMenu(); it.run();
    });
    m.querySelector('button').focus();
  }


  function modal({
    title,
    body,
    input,
    confirmText,
    danger,
    onConfirm,
    extraHtml = '',
    closeOnConfirm = true
  }) {
    const root = $('#modal-root');

    root.innerHTML = `
      <div class="overlay">
        <div class="modal"
            role="dialog"
            aria-modal="true"
            aria-label="${esc(title)}">

          <h2>${esc(title)}</h2>

          ${
            input
              ? `
                <label for="m-input">${esc(input.label)}</label>
                <input
                  id="m-input"
                  maxlength="60"
                  autocomplete="off"
                  value="${esc(input.value)}"
                >
                <div class="err" id="m-err"></div>
              `
              : ''
          }

          ${body ? `<p>${body}</p>` : ''}

          ${extraHtml}

          <div class="actions">
            <button class="btn" data-close>Cancel</button>
            <button class="btn ${danger ? 'danger' : 'primary'}" data-ok>
              ${esc(confirmText)}
            </button>
          </div>

        </div>
      </div>
    `;

    const field = $('#m-input');
    const okButton = root.querySelector('[data-ok]');
    const cancelButton = root.querySelector('[data-close]');

    let busy = false;

    const close = (force = false) => {
      if (busy && !force) return;

      busy = false;
      root.innerHTML = '';
      document.removeEventListener('keydown', onKey);
    };

    const setProgress = (stage, percentage) => {
      const progressBox = $('#upload-progress');

      if (!progressBox) return;

      progressBox.hidden = false;

      $('#upload-stage').textContent = stage;
      $('#upload-percent').textContent = `${percentage}%`;
      $('#upload-bar').style.width = `${percentage}%`;
    };

    const setError = (message) => {
      const errorElement = $('#m-err');

      if (errorElement) {
        errorElement.textContent = message;
      }

      const uploadError = $('#upload-error');

      if (uploadError) {
        uploadError.textContent = message;
      }
    };

    const ok = async () => {

      if (busy) return;

      let value;

      if (field) {
        value = field.value.trim();

        if (!value) {
          setError('Enter a name to continue.');
          field.focus();
          return;
        }
      }

      if (closeOnConfirm) {
        close();
        await onConfirm(value);
        return;
      }

      busy = true;

      okButton.disabled = true;
      cancelButton.disabled = true;

      okButton.textContent = 'Processing...';

      try {
        await onConfirm(value, {
          close,
          setProgress,
          setError
        });

      } catch (error) {

        setError(error.message || 'Something went wrong.');

        busy = false;

        okButton.disabled = false;
        cancelButton.disabled = false;

        okButton.textContent = confirmText;
      }
    };

    const onKey = e => {
      if (e.key === 'Escape') close();

      if (
        e.key === 'Enter' &&
        field &&
        document.activeElement === field
      ) {
        ok();
      }
    };

    document.addEventListener('keydown', onKey);

    okButton.addEventListener('click', ok);
    cancelButton.addEventListener('click', close);

    root.querySelector('.overlay').addEventListener(
      'mousedown',
      e => {
        if (e.target.classList.contains('overlay')) {
          close();
        }
      }
    );

    if (field) {
      field.focus();
      field.select();
    } else {
      okButton.focus();
    }
  }


  /* ---------- Sidebar (mobile) ---------- */
  const closeSidebar = () => document.body.classList.remove('sidebar-open');

  /* ---------- Events ---------- */
  document.addEventListener('click', e => {
    const k = e.target.closest('[data-menu]');
    if (k) {
      e.stopPropagation();
      const id = k.dataset.id;
      if (k.dataset.menu === 'project') {
        openMenu(k, [{ label: 'Edit', icon: icons.edit, run: () => renameProject(id) }, { label: 'Delete', icon: icons.trash, danger: true, run: () => deleteProject(id) }]);
      } else {
        openMenu(k, [{ label: 'Edit', icon: icons.edit, run: () => renameChat(id) }, { label: 'Delete', icon: icons.trash, danger: true, run: () => deleteChat(id) }]);
      }
      return;
    }
    if (!e.target.closest('#menu')) closeMenu();

    const a = e.target.closest('[data-action]');
    if (!a) return;
    const id = a.dataset.id;
    switch (a.dataset.action) {
    case 'select-answer': {
        const chat = activeChat();
        if (!chat || !chat.projectId) break;

        selectedAnswerId = a.dataset.msg;

        // DEBUG LOGS
        console.log("Selected answer ID:", selectedAnswerId);
        console.log(
            "Selected message:",
            activeChat()?.messages.find(m => m.id === selectedAnswerId)
        );

        renderDocumentPanel(chat.projectId);
        updateMessages(false);

        break;
      }
      case 'new-project': createProject(); break;
      case 'new-general-chat': newChat(null); break;
      case 'new-project-chat': newChat(id); break;
      case 'open-project': openProject(id); break;
      case 'open-chat': { const c = state.chats.find(x => x.id === id); if (c) selectChat(c); break; }
      case 'send': send(); break;
      case 'token-usage': {
        const popup = document.getElementById('token-popup');
        if (popup) popup.hidden = !popup.hidden;
        break;
      }
      case 'toggle-sidebar': document.body.classList.toggle('sidebar-open'); break;
      case 'close-sidebar': closeSidebar(); break;

    case 'fb': {
      const chat = activeChat();
      const m = chat && chat.messages.find(x => x.id === a.dataset.msg);

      if (!chat || !m || !m.responseId || m.feedback) break;

      const feedbackValue = a.dataset.v === 'up' ? 'good' : 'bad';

      apiRequest('/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          chat_id: chat.id,
          response_id: m.responseId,
          query: m.feedbackQuery || '',
          response: m.text,
          feedback: feedbackValue,
          context: JSON.stringify(m.feedbackContext || []),
          answer_source: m.answerSource || '',
          project_id: chat.projectId || null
        })
      })
        .then(() => {
          m.feedback = a.dataset.v;
          save();
          updateMessages(false);
        })
        .catch(error => {
          console.error('Feedback submission failed:', error);
          alert(error.message || 'Could not submit feedback.');
        });

      break;
    }
    case 'select-document': {
    selectDocument(
      a.dataset.projectId,
      a.dataset.filename
    );
    break;
  }

  case 'toggle-documents': {
    const projectId = a.dataset.projectId;

    if (collapsedDocumentLists.has(projectId)) {
      collapsedDocumentLists.delete(projectId);
    } else {
      collapsedDocumentLists.add(projectId);
    }

    renderDocumentPanel(projectId);
    break;
  }


    }
  });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeMenu();
    if (e.key === 'Enter' && e.target.matches && e.target.matches('.item[data-action], .chat-card[data-action]')) e.target.click();
  });
  window.addEventListener('resize', closeMenu);


  document.addEventListener('click', event => {
    const highlight = event.target.closest('.doc-highlight');
    if (!highlight) return;

    document.querySelectorAll('.doc-highlight.active')
      .forEach(element => element.classList.remove('active'));

    highlight.classList.add('active');

    highlight.scrollIntoView({
      behavior: 'smooth',
      block: 'center'
    });
  });


initialize();
})();

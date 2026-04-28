// ==UserScript==
// @name         Clawdi Memory Capture — Claude Web
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  Capture Claude web UI conversations to aaa-memory
// @author       You
// @match        https://claude.ai/*
// @grant        none
// ==/UserScript==

(function() {
    'use strict';

    const OUTPUT_DIR = '/home/misscheta/knowledge/raw/web/claude-web/';

    console.log('[Clawdi] Claude web capture loaded');

    // Claude.ai uses websockets primarily, but also REST for message submission
    // We'll capture via fetch interception and DOM observation
    const origFetch = window.fetch;
    window.fetch = async (...args) => {
        const [resource, options] = args;
        const result = await origFetch(...args);

        try {
            const url = resource.toString();
            if (url.includes('/api/') || url.includes('/messages')) {
                const cloned = result.clone();
                const body = await cloned.json();
                if (body?.messages || body?.urns) {
                    captureMessage(body, url);
                }
            }
        } catch (e) {}

        return result;
    };

    // Also observe DOM for rendered messages (fallback)
    const observer = new MutationObserver(() => {
        captureFromDOM();
    });
    observer.observe(document.body, { childList: true, subtree: true });

    function captureMessage(body, url) {
        // Claude API format varies — normalize
        const messages = body.messages || [];
        const turn = {
            timestamp: new Date().toISOString(),
            platform: 'claude-web',
            url: url,
            messages: messages.map(m => ({
                role: m.role,
                content: m.content,
                model: m.model || null
            }))
        };
        saveTurn(turn);
    }

    function captureFromDOM() {
        // Fallback: extract from Claude UI DOM
        const messages = [];
        document.querySelectorAll('.font-message, [data-message-id]').forEach(el => {
            const role = el.classList.contains('font-message-user') ? 'user' : 'model';
            const content = el.textContent?.trim() || '';
            if (content) messages.push({ role, content });
        });

        if (messages.length > 0) {
            const turn = {
                timestamp: new Date().toISOString(),
                platform: 'claude-web',
                url: window.location.href,
                messages
            };
            saveTurn(turn);
        }
    }

    function saveTurn(turn) {
        const filename = `${OUTPUT_DIR}${Date.now()}-${Math.random().toString(36).slice(2,8)}.jsonl`;
        const line = JSON.stringify(turn) + '\n';

        if (typeof GM_xmlhttpRequest !== 'undefined') {
            GM_xmlhttpRequest({
                method: 'POST',
                url: 'file://' + filename,
                data: line,
                headers: { 'Content-Type': 'application/json' }
            }).catch(() => { saveToLocalStorage(turn); });
        } else {
            saveToLocalStorage(turn);
        }
    }

    function saveToLocalStorage(turn) {
        try {
            const key = 'clawdi_capture_' + Date.now();
            localStorage.setItem(key, JSON.stringify(turn));
        } catch (e) {}
    }

    console.log('[Clawdi] Claude web capture initialized');
})();

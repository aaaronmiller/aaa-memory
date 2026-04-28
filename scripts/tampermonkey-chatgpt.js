// ==UserScript==
// @name         Clawdi Memory Capture — ChatGPT
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  Capture ChatGPT conversations to aaa-memory
// @author       You
// @match        https://chat.openai.com/*
// @match        https://chatgpt.com/*
// @grant        none
// ==/UserScript==

(function() {
    'use strict';

    const OUTPUT_DIR = '/home/misscheta/knowledge/raw/web/chatgpt/';
    const MAX_RETRIES = 3;

    // Ensure output directory exists
    if (typeof window !== 'undefined') {
        // Browser side — we'll use GM_setValue if available, else fallback
        console.log('[Clawdi] ChatGPT capture loaded');
    }

    // Intercept fetch
    const origFetch = window.fetch;
    window.fetch = async (...args) => {
        const [resource, options] = args;
        const result = await origFetch(...args);

        // Try to capture conversation turns
        try {
            const url = resource.toString();
            if (url.includes('/backend-api/') || url.includes('/conversation')) {
                // Clone response to read body without consuming original
                const cloned = result.clone();
                const body = await cloned.json();
                captureIfMessage(body, url);
            }
        } catch (e) {
            // silently ignore — capture is best-effort
        }

        return result;
    };

    function captureIfMessage(body, url) {
        // ChatGPT API responses contain messages array
        const messages = body?.messages || body?.data?.[0]?.messages || [];
        if (!messages.length) return;

        const turn = {
            timestamp: new Date().toISOString(),
            platform: 'chatgpt',
            url: url,
            messages: messages.map(m => ({
                role: m.role,
                content: m.content?.parts?.[0] || m.content?.text || String(m.content),
                model: m.model || null
            }))
        };

        saveTurn(turn);
    }

    function saveTurn(turn) {
        const filename = `${OUTPUT_DIR}${Date.now()}-${Math.random().toString(36).slice(2,8)}.jsonl`;
        const line = JSON.stringify(turn) + '\n';

        // Use GM_xmlhttpRequest for file write (Tampermonkey GM_* API)
        if (typeof GM_xmlhttpRequest !== 'undefined') {
            GM_xmlhttpRequest({
                method: 'POST',
                url: 'file://' + filename,
                data: line,
                headers: { 'Content-Type': 'application/json' }
            }).catch(() => {
                // file:// often blocked — fallback to localStorage
                saveToLocalStorage(turn);
            });
        } else {
            saveToLocalStorage(turn);
        }
    }

    function saveToLocalStorage(turn) {
        try {
            const key = 'clawdi_capture_' + Date.now();
            localStorage.setItem(key, JSON.stringify(turn));
            // Background script will flush to disk periodically
        } catch (e) {
            // Storage full — drop
        }
    }

    console.log('[Clawdi] ChatGPT capture initialized');
})();

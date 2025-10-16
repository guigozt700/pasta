// Roteamento simples
window.addEventListener('hashchange', navegar);
function navegar() {
  const rota = location.hash.replace('#', '') || 'home';
  document.querySelectorAll('.page').forEach(p => p.style.display = 'none');
  const el = document.getElementById(rota);
  if (el) el.style.display = 'block';
}
navegar();

// ===== Cifra Vigenère =====
const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

function normalize(s) {
  return s.toUpperCase().replace(/[^A-Z]/g, '');
}

function vigenereEncrypt(plain, key) {
  plain = normalize(plain);
  key = normalize(key);
  let out = '';
  for (let i = 0; i < plain.length; i++) {
    const p = alphabet.indexOf(plain[i]);
    const k = alphabet.indexOf(key[i % key.length]);
    const c = (p + k) % alphabet.length;
    out += alphabet[c];
  }
  return out;
}

function vigenereDecrypt(cipher, key) {
  cipher = normalize(cipher);
  key = normalize(key);
  let out = '';
  for (let i = 0; i < cipher.length; i++) {
    const c = alphabet.indexOf(cipher[i]);
    const k = alphabet.indexOf(key[i % key.length]);
    const p = (c - k + alphabet.length) % alphabet.length;
    out += alphabet[p];
  }
  return out;
}

// ===== Eventos da Cifra =====
document.getElementById('btnCifrar').addEventListener('click', () => {
  const key = document.getElementById('vkey').value;
  const pt = document.getElementById('plaintext').value;
  const result = vigenereEncrypt(pt, key);
  document.getElementById('cipherResult').textContent = result;
});

document.getElementById('btnDecifrar').addEventListener('click', () => {
  const key = document.getElementById('vkey').value;
  const ct = document.getElementById('plaintext').value;
  const result = vigenereDecrypt(ct, key);
  document.getElementById('cipherResult').textContent = result;
});

// ====== Geração de Dataset ======
function gerarDataset(n, maxLen, key) {
  const arr = [];
  for (let i = 0; i < n; i++) {
    let s = '';
    for (let j = 0; j < maxLen; j++)
      s += alphabet[Math.floor(Math.random() * alphabet.length)];
    arr.push({
      plain: s,
      cipher: vigenereEncrypt(s, key)
    });
  }
  return arr;
}

// ====== TensorFlow.js ======
let model = null;
let DATA = null;

function textToIndices(text, maxLen) {
  text = text.padEnd(maxLen, 'A').slice(0, maxLen);
  return Array.from(text).map(ch => alphabet.indexOf(ch));
}

function indicesToText(indices) {
  return indices.map(i => alphabet[i] || 'A').join('');
}

function oneHot(Y, maxLen, vocabSize) {
  const N = Y.length;
  const buffer = tf.buffer([N, maxLen, vocabSize]);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < maxLen; j++) {
      buffer.set(1, i, j, Y[i][j]);
    }
  }
  return buffer.toTensor();
}

function buildModel(maxLen, vocab) {
  const input = tf.input({ shape: [maxLen], dtype: 'int32' });
  const emb = tf.layers.embedding({ inputDim: vocab, outputDim: 16 }).apply(input);
  const flat = tf.layers.flatten().apply(emb);
  const hidden = tf.layers.dense({ units: 64, activation: 'relu' }).apply(flat);
  const out = tf.layers.dense({ units: maxLen * vocab, activation: 'softmax' }).apply(hidden);
  const reshaped = tf.layers.reshape({ targetShape: [maxLen, vocab] }).apply(out);
  const model = tf.model({ inputs: input, outputs: reshaped });
  model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy' });
  return model;
}

const logBox = document.getElementById('mlLogs');
function log(msg) {
  logBox.textContent += msg + "\n";
  logBox.scrollTop = logBox.scrollHeight;
}

document.getElementById('genDataBtn').addEventListener('click', () => {
  const n = parseInt(document.getElementById('dsSize').value);
  const maxLen = parseInt(document.getElementById('maxLen').value);
  const key = document.getElementById('genKey').value;
  DATA = gerarDataset(n, maxLen, key);
  log(`Dataset gerado com ${n} exemplos.`);
});

document.getElementById('trainBtn').addEventListener('click', async () => {
  if (!DATA) return alert('Gere o dataset primeiro!');
  const maxLen = parseInt(document.getElementById('maxLen').value);
  const vocab = alphabet.length;

  const X = DATA.map(d => textToIndices(d.cipher, maxLen));
  const Y = DATA.map(d => textToIndices(d.plain, maxLen));

  const xs = tf.tensor2d(X, [X.length, maxLen], 'int32');
  const ys = oneHot(Y, maxLen, vocab);

  model = buildModel(maxLen, vocab);
  log("Treinando modelo...");

  await model.fit(xs, ys, {
    epochs: 10,
    batchSize: 64,
    validationSplit: 0.1,
    callbacks: {
      onEpochEnd: (ep, logs) => log(`Época ${ep + 1} — loss: ${logs.loss.toFixed(4)}`)
    }
  });

  log("Treino concluído!");
});

document.getElementById('evalBtn').addEventListener('click', () => {
  if (!model) return alert('Treine o modelo primeiro!');
  const maxLen = parseInt(document.getElementById('maxLen').value);
  const sample = DATA.slice(0, 10);
  const X = sample.map(d => textToIndices(d.cipher, maxLen));
  const xs = tf.tensor2d(X, [X.length, maxLen], 'int32');
  const preds = model.predict(xs);
  const predsIdx = preds.argMax(-1).arraySync();
  predsIdx.forEach((arr, i) => {
    const dec = indicesToText(arr);
    log(`Cipher: ${sample[i].cipher} → Pred: ${dec}`);
  });
});

document.getElementById('predictBtn').addEventListener('click', () => {
  if (!model) return alert('Treine o modelo primeiro!');
  const c = document.getElementById('testCipher').value;
  const maxLen = parseInt(document.getElementById('maxLen').value);
  const x = tf.tensor2d([textToIndices(c, maxLen)], [1, maxLen], 'int32');
  const pred = model.predict(x);
  const arr = pred.argMax(-1).arraySync()[0];
  const dec = indicesToText(arr);
  document.getElementById('predictOut').textContent = dec;
});

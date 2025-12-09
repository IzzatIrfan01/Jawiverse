import React, { useState, useEffect } from 'react';
import { 
  XCircle, 
  Clipboard, 
  Copy, 
  Volume2, 
  RefreshCw 
} from 'react-feather';  // Specific icons imported

const Converter = ({ currentLang }) => {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [currentMode, setCurrentMode] = useState('jawi-roman');
  const [isLoading, setIsLoading] = useState(false);

  const translations = {
    en: {
      modeTitle: "Select Transliteration Mode",
      mode1: "Classical Jawi → Romanized Malay",
      mode2: "Romanized Malay → Classical Jawi",
      mode3: "Modern Jawi → Romanized Malay",
      mode4: "Romanized Malay → Modern Jawi",
      inputLabel: "Input Text",
      outputLabel: "Output Text",
      outputPlaceholder: "Transliterated text will appear here...",
      clear: "Clear",
      sample: "Sample",
      char: "characters",
      copy: "Copy",
      speak: "Speak",
      transliterate: "Transliterate",
    },
    ms: {
      modeTitle: "Pilih Mod Transliterasi",
      mode1: "Jawi Klasik → Melayu Rumi",
      mode2: "Melayu Rumi → Jawi Klasik",
      mode3: "Jawi Moden → Melayu Rumi",
      mode4: "Melayu Rumi → Jawi Moden",
      inputLabel: "Teks Input",
      outputLabel: "Teks Output",
      outputPlaceholder: "Teks transliterasi akan dipaparkan di sini...",
      clear: "Kosongkan",
      sample: "Contoh",
      char: "aksara",
      copy: "Salin",
      speak: "Dengar",
      transliterate: "Transliterasi",
    }
  };

  const t = translations[currentLang];

  const modes = [
    { id: 'jawi-roman', label: t.mode1 },
    { id: 'roman-jawi', label: t.mode2 },
    { id: 'modern-jawi-roman', label: t.mode3 },
    { id: 'roman-modern-jawi', label: t.mode4 }
  ];

  const getInputPlaceholder = () => {
    if (currentMode.includes('jawi-')) {
      return currentLang === 'en' 
        ? "Enter Jawi text here..."
        : "Masukkan teks Jawi di sini...";
    } else {
      return currentLang === 'en'
        ? "Enter Romanized Malay text here..."
        : "Masukkan teks Melayu Rumi di sini...";
    }
  };

  const handleSampleText = () => {
    let sampleText = '';
    if (currentMode.includes('jawi')) {
      sampleText = 'كيت ممنوع جوݢ ماسق وقت ڤاعين تيدق ڤرناه';
    } else {
      sampleText = 'Kita dilarang masuk ketika permainan sedang berlangsung';
    }
    setInputText(sampleText);
  };

  const handleClearInput = () => {
    setInputText('');
  };

  const handleCopyOutput = () => {
    navigator.clipboard.writeText(outputText);
    alert(currentLang === 'en' ? 'Copied to clipboard!' : 'Disalin ke papan keratan!');
  };

  const handleSpeakOutput = () => {
    if (!outputText || outputText.includes('Transliterated') || outputText.includes('transliterasi')) {
      alert(currentLang === 'en' ? 'No output to speak!' : 'Tiada output untuk diucapkan!');
      return;
    }

    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(outputText);
      if (outputText.match(/[\u0600-\u06FF]/)) {
        utterance.lang = 'ms';
      } else {
        utterance.lang = currentLang === 'en' ? 'en-US' : 'ms-MY';
      }
      speechSynthesis.speak(utterance);
    }
  };

  const handleTransliterate = () => {
    if (!inputText.trim()) {
      alert(currentLang === 'en' ? 'Please enter some text!' : 'Sila masukkan teks!');
      return;
    }

    setIsLoading(true);

    setTimeout(() => {
      let result = '';
      if (inputText.includes('كيت') || inputText.includes('Kita')) {
        result = inputText.includes('كيت') 
          ? 'Kita dilarang masuk ketika permainan sedang berlangsung'
          : 'كيت ممنوع جوݢ ماسق وقت ڤاعين تيدق ڤرناه';
      } else {
        result = currentLang === 'en' 
          ? 'Example transliterated output would appear here. This is a simulation.'
          : 'Contoh output transliterasi akan muncul di sini. Ini adalah simulasi.';
      }
      setOutputText(result);
      setIsLoading(false);
    }, 1000);
  };

  return (
    <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden">
      {/* Mode Selection */}
      <div className="bg-gradient-to-r from-primary to-secondary p-6">
        <h3 className="text-white text-xl font-semibold mb-4">
          {t.modeTitle}
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {modes.map((mode) => (
            <button
              key={mode.id}
              className={`py-3 px-6 rounded-lg font-medium transition-all ${
                currentMode === mode.id
                  ? 'bg-white text-primary ring-2 ring-secondary'
                  : 'bg-white/90 text-primary hover:bg-white'
              }`}
              onClick={() => setCurrentMode(mode.id)}
            >
              {mode.label}
            </button>
          ))}
        </div>
      </div>

      {/* Converter */}
      <div className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Input Section */}
          <div>
            <label htmlFor="input-text" className="block text-gray-700 font-medium mb-2">
              {t.inputLabel}
            </label>
            <textarea
              id="input-text"
              rows="8"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
              placeholder={getInputPlaceholder()}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />
            <div className="flex justify-between mt-2">
              <div className="flex items-center space-x-2">
                <button
                  onClick={handleClearInput}
                  className="text-gray-500 hover:text-primary flex items-center space-x-1"
                >
                  <XCircle size={16} />  {/* Use XCircle component directly */}
                  <span>{t.clear}</span>
                </button>
                <button
                  onClick={handleSampleText}
                  className="text-gray-500 hover:text-primary flex items-center space-x-1"
                >
                  <Clipboard size={16} />  {/* Use Clipboard component directly */}
                  <span>{t.sample}</span>
                </button>
              </div>
              <span className="text-gray-500 text-sm">
                {inputText.length} {t.char}
              </span>
            </div>
          </div>

          {/* Output Section */}
          <div>
            <label className="block text-gray-700 font-medium mb-2">
              {t.outputLabel}
            </label>
            <div className="w-full px-4 py-3 min-h-[200px] border border-gray-300 rounded-lg bg-gray-50 text-jawi">
              {outputText ? (
                <p className="text-gray-800">{outputText}</p>
              ) : (
                <p className="text-gray-400">{t.outputPlaceholder}</p>
              )}
            </div>
            <div className="flex justify-between mt-2">
              <button
                onClick={handleCopyOutput}
                className="text-gray-500 hover:text-primary flex items-center space-x-1"
              >
                <Copy size={16} />  {/* Use Copy component directly */}
                <span>{t.copy}</span>
              </button>
              <button
                onClick={handleSpeakOutput}
                className="text-gray-500 hover:text-primary flex items-center space-x-1"
              >
                <Volume2 size={16} />  {/* Use Volume2 component directly */}
                <span>{t.speak}</span>
              </button>
            </div>
          </div>
        </div>

        {/* Transliterate Button */}
        <div className="mt-6 flex justify-center">
          <button
            onClick={handleTransliterate}
            disabled={isLoading}
            className="bg-primary hover:bg-[#6E4429] text-white font-bold py-3 px-8 rounded-lg shadow-lg transform hover:scale-105 transition duration-300 flex items-center space-x-2 disabled:opacity-50"
          >
            {isLoading && (
              <RefreshCw className="animate-spin" /> 
            )}
            <span>{t.transliterate}</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default Converter;
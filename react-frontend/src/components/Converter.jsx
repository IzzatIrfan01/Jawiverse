import React, { useState } from 'react';
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
  const [currentMode, setCurrentMode] = useState('old-jawi-roman');
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
    { id: 'old-jawi-roman', label: t.mode1 },
    { id: 'roman-old-jawi', label: t.mode2 },
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

    // Check for the specific mode to determine the exact sample
    switch (currentMode) {
        case 'roman-modern-jawi':
            // Sample for Rumi input (modern-style translation)
            sampleText = 'Ini hasil bergadang dan kerja keras';
            break;
            
        case 'old-jawi-roman':
            // Sample for Jawi input (classic Jawi text that translates to Rumi)
            sampleText = 'دان‭ ‬جوݢ‭ ‬كرج‮٢‬‭ ‬مڠوكير‭ ‬دان‭ ‬ممبواة‭ ‬ڤركاكس‮٢‬‭ ‬لاࢨن‭ ‬داڤة‭ ‬موده‭ ‬دحاصلكن‭.‬';
            break;
        case 'modern-jawi-roman': 
            // Sample for Jawi input (classic Jawi text that translates to Rumi)
            sampleText = 'اين حاصيل برڬادڠ دان كرجا كراس';
            break;

        case 'roman-old-jawi':
            // Sample for Rumi input (classic Jawi translation)
            // You may want to define a new Rumi sample here, or use the modern one for now.
            // Using the modern Rumi one since the output is the only difference.
            sampleText = 'Dan juga kerja-kerja mengukir dan membuat perkakas-perkakas lain dapat mudah dihasilkan.';
            break;

        default:
            // Fallback sample if a mode is not recognized (optional)
            sampleText = ''; 
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

    // Define the specific sample input and output pairs for the simulation
    const romanModernSampleInput = 'Ini hasil bergadang dan kerja keras';
    const jawiModernSampleOutput = 'اين حاصيل برڬادڠ دان كرجا كراس'; // Output for roman-modern-jawi

    const jawiClassicSampleInput = 'دان‭ ‬جوݢ‭ ‬كرج‮٢‬‭ ‬مڠوكير‭ ‬دان‭ ‬ممبواة‭ ‬ڤركاكس‮٢‬‭ ‬لاࢨن‭ ‬داڤة‭ ‬موده‭ ‬دحاصلكن‭.‬';
    const romanClassicSampleOutput = 'Dan juga kerja-kerja mengukir dan membuat perkakas-perkakas lain dapat mudah dihasilkan.'; // Output for old-jawi-roman

    let result = '';

    // Simulate Transliteration based on current mode and input text
    switch (currentMode) {
      // Rumi Input Modes (Output is Jawi)
      case 'roman-modern-jawi':
        if (inputText.trim() === romanModernSampleInput.trim()) {
          result = jawiModernSampleOutput;
        }
        break;

      case 'roman-old-jawi':
        if (inputText.trim() === romanClassicSampleOutput.trim()) {
          result = jawiClassicSampleInput;
        }
        break;

      // Jawi Input Modes (Output is Rumi)
      case 'old-jawi-roman':
        if (inputText.trim() === jawiClassicSampleInput.trim()) {
          result = romanClassicSampleOutput;
        }
        break;

      case 'modern-jawi-roman':
        if (inputText.trim() === jawiModernSampleOutput.trim()) {
          result = romanModernSampleInput;
        }
        break;

      default:
        // No specific mode matched — leave `result` empty so the generic
        // fallback below sets the simulated output.
        break;
    }


    // If no specific sample match, use the generic simulation text
    if (!result) {
        result = currentLang === 'en' 
            ? 'Transliterated output would appear here. This is a simulation.'
            : 'Hasil transliterasi akan muncul di sini. Ini adalah simulasi.';
    }

    // Delay simulation
    setTimeout(() => {
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
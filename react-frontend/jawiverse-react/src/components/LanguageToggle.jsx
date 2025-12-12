import React from 'react';
import { translations } from '../constants/translations';

const LanguageToggle = ({ currentLang, onLanguageChange }) => {
  return (
    <div className="language-toggle mr-4">
      <button
        id="lang-en"
        className={`px-4 py-2 rounded-full transition-all ${
          currentLang === 'en'
            ? 'bg-white text-primary shadow-lg'
            : 'bg-transparent text-gray-600 hover:text-secondary'
        }`}
        onClick={() => onLanguageChange('en')}
      >
        English
      </button>
      <button
        id="lang-ms"
        className={`px-4 py-2 rounded-full transition-all ${
          currentLang === 'ms'
            ? 'bg-white text-primary shadow-lg'
            : 'bg-transparent text-gray-600 hover:text-secondary'
        }`}
        onClick={() => onLanguageChange('ms')}
      >
        Bahasa Melayu
      </button>
    </div>
  );
};

export default LanguageToggle;
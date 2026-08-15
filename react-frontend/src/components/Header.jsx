import React from 'react';
import { AlignLeft, Menu } from 'react-feather';  // Specific icons imported
import LanguageToggle from './LanguageToggle';

const Header = ({ currentLang, onLanguageChange }) => {
  const translations = {
    en: {
      siteTitle: "The Digital Key to Jawi Heritage",
      siteSubtitle: "Bridging classical Jawi and modern Malay with AI-powered precision",
      navHome: "Home",
      navAbout: "About",
      navGuide: "Guide"
    },
    ms: {
      siteTitle: "Kunci Digital kepada Warisan Jawi",
      siteSubtitle: "Menghubungkan tulisan Jawi klasik dan Melayu moden dengan ketepatan berkuasa AI",
      navHome: "Laman Utama",
      navAbout: "Tentang",
      navGuide: "Panduan"
    }
  };

  const t = translations[currentLang];

  return (
    <header className="bg-primary text-white shadow-lg">
      <div className="container mx-auto px-4 py-6">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <AlignLeft className="text-2xl" />  {/* Use AlignLeft component directly */}
            <h1 className="text-2xl font-bold">JawiVerse</h1>
          </div>
          <div className="flex items-center space-x-4">

            <nav className="hidden md:flex space-x-8">
              <a href="/home" className="hover:text-secondary font-medium">
                {t.navHome}
              </a>
              <a href="/about" className="hover:text-secondary font-medium">
                {t.navAbout}
              </a>
              <a href="/guide" className="hover:text-secondary font-medium">
                {t.navGuide}
              </a>
            </nav>
            <LanguageToggle 
              currentLang={currentLang} 
              onLanguageChange={onLanguageChange} 
            />
            <button className="md:hidden">
              <Menu />  {/* Use Menu component directly */}
            </button>
          </div>
        </div>
        <div className="mt-8 text-center">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            {t.siteTitle}
          </h2>
          <p className="text-xl opacity-90 max-w-2xl mx-auto">
            {t.siteSubtitle}
          </p>
        </div>
      </div>
    </header>
  );
};

export default Header;
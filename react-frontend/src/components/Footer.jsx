import React from 'react';
import { Twitter, GitHub, Mail } from 'react-feather';  // Note: GitHub not Github

const Footer = ({ currentLang }) => {
  const translations = {
    en: {
      footerTitle: "JawiVerse",
      footerDesc: "Bridging the past and present through script transformation technology.",
      footerResources: "Resources",
      footerLegal: "Legal",
      footerConnect: "Connect",
      footerDoc: "Documentation",
      footerApi: "API Reference",
      footerGuide: "Jawi Guide",
      footerPrivacy: "Privacy Policy",
      footerTerms: "Terms of Service",
      copyright: "© 2025 JawiVerse. All rights reserved."
    },
    ms: {
      footerTitle: "JawiVerse",
      footerDesc: "Menghubungkan masa lalu dan masa kini melalui teknologi transformasi tulisan.",
      footerResources: "Sumber",
      footerLegal: "Undang-undang",
      footerConnect: "Hubungi",
      footerDoc: "Dokumentasi",
      footerApi: "Rujukan API",
      footerGuide: "Panduan Jawi",
      footerPrivacy: "Dasar Privasi",
      footerTerms: "Terma Perkhidmatan",
      copyright: "© 2025 JawiVerse. Hak cipta terpelihara."
    }
  };

  const t = translations[currentLang];

  return (
    <footer className="bg-gray-800 text-white py-12">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Company Info */}
          <div>
            <h3 className="font-bold text-lg mb-4">{t.footerTitle}</h3>
            <p className="text-gray-400">{t.footerDesc}</p>
          </div>

          {/* Resources */}
          <div>
            <h3 className="font-bold text-lg mb-4">{t.footerResources}</h3>
            <ul className="space-y-2">
              <li>
                <a href="/doc" className="text-gray-400 hover:text-white">
                  {t.footerDoc}
                </a>
              </li>
              <li>
                <a href="/api" className="text-gray-400 hover:text-white">
                  {t.footerApi}
                </a>
              </li>
              <li>
                <a href="/guide" className="text-gray-400 hover:text-white">
                  {t.footerGuide}
                </a>
              </li>
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h3 className="font-bold text-lg mb-4">{t.footerLegal}</h3>
            <ul className="space-y-2">
              <li>
                <a href="/privacy" className="text-gray-400 hover:text-white">
                  {t.footerPrivacy}
                </a>
              </li>
              <li>
                <a href="/footer" className="text-gray-400 hover:text-white">
                  {t.footerTerms}
                </a>
              </li>
            </ul>
          </div>

          {/* Connect */}
          <div>
            <h3 className="font-bold text-lg mb-4">{t.footerConnect}</h3>
            <div className="flex space-x-4">
              <a href="/twitter" className="text-gray-400 hover:text-white">
                <Twitter />  {/* Twitter component */}
              </a>
              <a href="/github" className="text-gray-400 hover:text-white">
                <GitHub />   {/* GitHub component with capital H */}
              </a>
              <a href="/email" className="text-gray-400 hover:text-white">
                <Mail />     {/* Mail component */}
              </a>
            </div>
          </div>
        </div>

        {/* Copyright */}
        <div className="border-t border-gray-700 mt-8 pt-8 text-center text-gray-400">
          <p>{t.copyright}</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
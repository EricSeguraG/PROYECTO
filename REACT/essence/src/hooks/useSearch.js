import { useState } from 'react';
import { perfumeAPI } from '../services/api';
import celebritiesDB from '../data/celebritiesDB';

export const useSearch = () => {
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchType, setSearchType] = useState('general');
  const [loadingImages, setLoadingImages] = useState(false);

  // Función para normalizar texto
  const normalizeText = (text) => {
    if (!text) return '';
    return text
      .toLowerCase()
      .normalize('NFD').replace(/[\u0300-\u036f]/g, '')
      .replace(/[^\w\s]/gi, '')
      .trim();
  };

  // Mapeo de nombres comunes
  const perfumeNameMapping = {
    // Creed
    "Creed Aventus": "Aventus",
    "Aventus": "Aventus",
    "Creed Green Irish Tweed": "Green Irish Tweed",
    "Green Irish Tweed": "Green Irish Tweed",
    "Creed Bois du Portugal": "Bois du Portugal",
    "Creed Silver Mountain Water": "Silver Mountain Water",
    "Creed Millésime Impérial": "Millésime Impérial",
    "Creed Original Santal": "Original Santal",
    "Creed Viking": "Viking",
    
    // Tom Ford
    "Tom Ford Tobacco Vanille": "Tobacco Vanille",
    "Tobacco Vanille": "Tobacco Vanille",
    "Tom Ford Noir Extreme": "Noir Extreme",
    "Noir Extreme": "Noir Extreme",
    "Tom Ford Black Orchid": "Black Orchid",
    "Black Orchid": "Black Orchid",
    "Tom Ford Velvet Orchid": "Velvet Orchid",
    "Velvet Orchid": "Velvet Orchid",
    "Tom Ford Santal Blush": "Santal Blush",
    "Santal Blush": "Santal Blush",
    "Tom Ford Oud Wood": "Oud Wood",
    "Oud Wood": "Oud Wood",
    "Tom Ford Lost Cherry": "Lost Cherry",
    "Lost Cherry": "Lost Cherry",
    "Tom Ford Ombré Leather": "Ombré Leather",
    "Tom Ford Ombre Leather": "Ombré Leather",
    "Ombré Leather": "Ombré Leather",
    
    // Chanel
    "Chanel Coco Mademoiselle": "Coco Mademoiselle",
    "Coco Mademoiselle": "Coco Mademoiselle",
    "Bleu de Chanel": "Bleu de Chanel",
    "Chanel Chance Eau Tendre": "Chance Eau Tendre",
    "Chance Eau Tendre": "Chance Eau Tendre",
    "Chanel N°5 L'Eau": "N°5 L'Eau",
    "N°5 L'Eau": "N°5 L'Eau",
    
    // Dior
    "Dior Sauvage": "Sauvage",
    "Sauvage": "Sauvage",
    "Dior Homme Intense": "Homme Intense",
    "Homme Intense": "Homme Intense",
    "Dior Fahrenheit": "Fahrenheit",
    "Fahrenheit": "Fahrenheit",
    "Dior Homme Sport": "Homme Sport",
    "Homme Sport": "Homme Sport",
    "Dior J'adore": "J'adore",
    "J'adore": "J'adore",
    "Miss Dior": "Miss Dior",
    "Dior Sauvage Elixir": "Sauvage Elixir",
    "Sauvage Elixir": "Sauvage Elixir",
    
    // YSL
    "YSL Libre": "Libre",
    "Yves Saint Laurent Libre": "Libre",
    "Libre": "Libre",
    "Black Opium": "Black Opium",
    
    // Carolina Herrera
    "Good Girl": "Good Girl",
    
    // Jo Malone
    "Jo Malone Wood Sage & Sea Salt": "Wood Sage & Sea Salt",
    "Wood Sage & Sea Salt": "Wood Sage & Sea Salt",
    "Jo Malone English Pear & Freesia": "English Pear & Freesia",
    "English Pear & Freesia": "English Pear & Freesia",
    "Jo Malone Lime Basil & Mandarin": "Lime Basil & Mandarin",
    "Lime Basil & Mandarin": "Lime Basil & Mandarin",
    "Jo Malone Peony & Blush Suede": "Peony & Blush Suede",
    "Peony & Blush Suede": "Peony & Blush Suede",
    
    // Gucci
    "Gucci Mémoire d'une Odeur": "Mémoire d'une Odeur",
    "Gucci Memoire d'une Odeur": "Mémoire d'une Odeur",
    "Mémoire d'une Odeur": "Mémoire d'une Odeur",
    
    // Más
    "Acqua di Gio": "Acqua di Gio",
    "Giorgio Armani Acqua di Gio": "Acqua di Gio",
    "Acqua di Gio Profumo": "Acqua di Gio Profumo",
    "Acqua di Gio Profondo": "Acqua di Gio Profondo",
    "Terre d'Hermès": "Terre d'Hermès",
    "Paco Rabanne 1 Million": "1 Million",
    "Le Labo Santal 33": "Santal 33",
    "Prada L'Homme": "L'Homme",
    "Maison Francis Kurkdjian Baccarat Rouge 540": "Baccarat Rouge 540",
    "Maison Margiela REPLICA Jazz Club": "Jazz Club",
    "Lancôme La Vie Est Belle": "La Vie Est Belle",
    "Valentino Voce Viva": "Voce Viva",
    "Hugo Boss Boss Bottled": "Boss Bottled",
    "Fenty Eau de Parfum": "Fenty",
    "Kilian Love, don't be shy": "Love, don't be shy",
    "Glow by JLo": "Glow",
    "Lady Gaga Fame": "Fame",
    "S by Shakira": "S",
    "Billie Eilish Eilish": "Eilish",
    "Selena Gomez Rare": "Rare",
    "KKW Crystal Gardenia": "Crystal Gardenia",
    "Omnia": "Omnia",
  };

  // Función para normalizar nombre de perfume
  const normalizePerfumeName = (name) => {
    if (!name || typeof name !== 'string') return '';
    
    const trimmedName = name.trim();
    
    // 1. Verificar mapeo directo
    if (perfumeNameMapping[trimmedName]) {
      return perfumeNameMapping[trimmedName];
    }
    
    // 2. Verificar si el nombre contiene alguna clave del mapeo
    for (const [key, value] of Object.entries(perfumeNameMapping)) {
      if (trimmedName.toLowerCase().includes(key.toLowerCase()) || 
          key.toLowerCase().includes(trimmedName.toLowerCase())) {
        return value;
      }
    }
    
    // 3. Eliminar marcas comunes
    const cleanName = trimmedName
      .replace(/^(Creed|Tom Ford|Chanel|Dior|YSL|Yves Saint Laurent|Jo Malone|Gucci|Armani|Hermès|Prada|Le Labo|Maison Margiela|Lancôme|Valentino|Hugo Boss|Kilian|Ariana Grande|Lady Gaga|Selena Gomez|Billie Eilish|Shakira|Jennifer Lopez|Fenty|KKW)\s+/i, '')
      .trim();
    
    return cleanName || trimmedName;
  };

  // Función para obtener variaciones del nombre
  const getPerfumeVariations = (name) => {
    if (!name) return [''];
    
    const variations = [name];
    const normalized = normalizePerfumeName(name);
    
    if (normalized && normalized !== name) {
      variations.push(normalized);
    }
    
    // Añadir versión sin marca
    const withoutBrand = name.replace(/^(Creed|Tom Ford|Chanel|Dior|YSL|Yves Saint Laurent|Jo Malone|Gucci)\s+/i, '').trim();
    if (withoutBrand && withoutBrand !== name && !variations.includes(withoutBrand)) {
      variations.push(withoutBrand);
    }
    
    return variations;
  };

  // Función MEJORADA para buscar perfume en la API
  const searchPerfumeInAPI = async (perfumeName, brand) => {
    try {
      console.log(`🔍 Buscando: "${perfumeName}" (${brand})`);
      
      // Obtener variaciones del nombre
      const nameVariations = getPerfumeVariations(perfumeName);
      
      // Buscar perfumes por marca
      let perfumes = [];
      let page = 1;
      
      // Buscar en 2 páginas máximo para ser rápido
      while (perfumes.length < 50 && page <= 2) {
        const pageResults = await perfumeAPI.getPerfumesByBrand(brand, page);
        if (!pageResults || pageResults.length === 0) break;
        
        perfumes = [...perfumes, ...pageResults];
        page++;
      }
      
      if (perfumes.length === 0) {
        return null;
      }
      
      let foundPerfume = null;
      let bestScore = 0;
      
      // Buscar entre todas las variaciones
      for (const variation of nameVariations) {
        const normalizedVariation = normalizeText(variation);
        
        for (const perfume of perfumes) {
          const apiName = perfume.perfume || perfume.nombre || '';
          const normalizedApiName = normalizeText(apiName);
          
          // Coincidencia exacta
          if (normalizedApiName === normalizedVariation) {
            console.log(`✅ Encontrado: "${apiName}"`);
            return perfume;
          }
          
          // Uno contiene al otro
          if (normalizedApiName.includes(normalizedVariation) || 
              normalizedVariation.includes(normalizedApiName)) {
            const score = normalizedApiName.includes(normalizedVariation) ? 0.9 : 0.8;
            if (score > bestScore) {
              foundPerfume = perfume;
              bestScore = score;
            }
          }
          
          // Coincidencia de palabras clave
          const variationWords = normalizedVariation.split(/\s+/);
          const apiWords = normalizedApiName.split(/\s+/);
          let wordMatches = 0;
          
          variationWords.forEach(vWord => {
            if (vWord.length > 2 && apiWords.some(aWord => aWord.includes(vWord) || vWord.includes(aWord))) {
              wordMatches++;
            }
          });
          
          const wordScore = wordMatches / Math.max(variationWords.length, 1);
          if (wordScore > bestScore && wordScore > 0.5) {
            foundPerfume = perfume;
            bestScore = wordScore;
          }
        }
        
        // Si encontramos buena coincidencia, salir
        if (bestScore > 0.8) break;
      }
      
      if (foundPerfume) {
        console.log(`✅ Encontrado: "${foundPerfume.perfume || foundPerfume.nombre}"`);
        return foundPerfume;
      }
      
      return null;
      
    } catch (error) {
      console.error(`Error buscando ${perfumeName} de ${brand}:`, error);
      return null;
    }
  };

  // Función para obtener imagen de perfume
  const getPerfumeImage = (url) => {
    if (!url) return null;
    
    const match = url.match(/(\d+)\.html$/);
    if (!match) return null;
    
    const id = match[1];
    return `https://fimgs.net/mdimg/perfume/375x500.${id}.jpg`;
  };

  // --- BÚSQUEDA DE CELEBRITIES (SIMPLIFICADA) ---
  const searchByCelebrity = async (celebrityName) => {
    if (!celebrityName || !celebrityName.trim()) {
      setSearchResults([]);
      return;
    }

    setLoading(true);
    setLoadingImages(false);
    setError(null);
    setSearchType('celebrities');
    
    try {
      // 1. Buscar en base de datos local
      const searchTerm = normalizeText(celebrityName);
      const celebrityResults = celebritiesDB.filter(celebrity => 
        normalizeText(celebrity.celebrity).includes(searchTerm) ||
        searchTerm.includes(normalizeText(celebrity.celebrity))
      );
      
      if (celebrityResults.length === 0) {
        setError(`No encontramos a "${celebrityName}" en nuestra base de datos.`);
        setSearchResults([]);
        setLoading(false);
        return;
      }
      
      // 2. Mostrar resultados iniciales
      const initialResults = celebrityResults.map(celebrity => ({
        ...celebrity,
        perfumes: celebrity.perfumes.map(perfume => ({
          ...perfume,
          img: null
        }))
      }));
      
      setSearchResults(initialResults);
      setLoading(false);
      setLoadingImages(true);
      
      // 3. Buscar imágenes en segundo plano (simplificado)
      const resultsWithImages = await Promise.all(
        celebrityResults.map(async (celebrity) => {
          const perfumesWithImages = await Promise.all(
            celebrity.perfumes.map(async (perfume) => {
              const apiPerfume = await searchPerfumeInAPI(perfume.name, perfume.brand);
              
              let imageUrl = null;
              
              if (apiPerfume && apiPerfume.url) {
                imageUrl = getPerfumeImage(apiPerfume.url);
              }
              
              return {
                ...perfume,
                img: imageUrl
              };
            })
          );
          
          return {
            ...celebrity,
            perfumes: perfumesWithImages
          };
        })
      );
      
      setSearchResults(resultsWithImages);
      setLoadingImages(false);
      
    } catch (err) {
      console.error("Error buscando celebrity:", err);
      setError("Ocurrió un error al intentar buscar el famoso.");
      setSearchResults([]);
      setLoading(false);
      setLoadingImages(false);
    }
  };

  // --- BÚSQUEDA GENERAL DE PERFUMES ---
  const searchPerfumes = async (query, filters = {}) => {
    if (!query.trim() && 
        !filters.perfume && 
        !filters.marca && 
        !filters.genero && 
        !filters.nota && 
        !filters.acorde && 
        !filters.perfumista) {
      setSearchResults([]);
      return;
    }

    setLoading(true);
    setError(null);
    setSearchType('general');
    
    try {
      const results = await perfumeAPI.searchPerfumes(query, filters);
      setSearchResults(results);
    } catch (err) {
      setError(err.message);
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };

  // --- BÚSQUEDA DE SIMILARES ---
  const searchSimilar = async (perfumeName) => {
    setLoading(true);
    setError(null);
    setSearchType('general');
    
    try {
      const results = await perfumeAPI.searchSimilar(perfumeName);
      setSearchResults(results);
    } catch (err) {
      setError(err.message);
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };

  // --- BÚSQUEDA DE CLONES ---
  const searchClones = async (originalPerfumeId) => {
    setLoading(true);
    setError(null);
    setSearchType('clones');
    
    try {
      const results = await perfumeAPI.searchClones(originalPerfumeId);
      setSearchResults(results);
    } catch (err) {
      setError(err.message);
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };

  // --- LIMPIAR RESULTADOS ---
  const clearResults = () => {
    setSearchResults([]);
    setError(null);
    setSearchType('general');
    setLoading(false);
    setLoadingImages(false);
  };

  return {
    searchResults,
    loading,
    loadingImages,
    error,
    searchType,
    searchPerfumes,
    searchSimilar,
    searchClones,
    searchByCelebrity,
    clearResults
  };
};
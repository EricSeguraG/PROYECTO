// src/utils/perfumeNameMapper.js

// Mapeo de nombres comunes que varían entre celebritiesDB.js y la API
const perfumeNameMapping = {
  // === CREED ===
  "Creed Aventus": "Aventus",
  "Aventus": "Aventus",
  "Creed Green Irish Tweed": "Green Irish Tweed",
  "Green Irish Tweed": "Green Irish Tweed",
  "Creed Bois du Portugal": "Bois du Portugal",
  "Creed Silver Mountain Water": "Silver Mountain Water",
  "Creed Millésime Impérial": "Millésime Impérial",
  "Creed Original Santal": "Original Santal",
  "Creed Viking": "Viking",
  
  // === TOM FORD ===
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
  
  // === CHANEL ===
  "Chanel Coco Mademoiselle": "Coco Mademoiselle",
  "Coco Mademoiselle": "Coco Mademoiselle",
  "Bleu de Chanel": "Bleu de Chanel",
  "Chanel Chance Eau Tendre": "Chance Eau Tendre",
  "Chance Eau Tendre": "Chance Eau Tendre",
  "Chanel N°5 L'Eau": "N°5 L'Eau",
  "N°5 L'Eau": "N°5 L'Eau",
  
  // === DIOR ===
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
  
  // === YSL / YVES SAINT LAURENT ===
  "YSL Libre": "Libre",
  "Yves Saint Laurent Libre": "Libre",
  "Libre": "Libre",
  "Black Opium": "Black Opium",
  
  // === CAROLINA HERRERA ===
  "Good Girl": "Good Girl",
  
  // === JO MALONE ===
  "Jo Malone Wood Sage & Sea Salt": "Wood Sage & Sea Salt",
  "Wood Sage & Sea Salt": "Wood Sage & Sea Salt",
  "Jo Malone English Pear & Freesia": "English Pear & Freesia",
  "English Pear & Freesia": "English Pear & Freesia",
  "Jo Malone Lime Basil & Mandarin": "Lime Basil & Mandarin",
  "Lime Basil & Mandarin": "Lime Basil & Mandarin",
  "Jo Malone Peony & Blush Suede": "Peony & Blush Suede",
  "Peony & Blush Suede": "Peony & Blush Suede",
  
  // === GUCCI ===
  "Gucci Mémoire d'une Odeur": "Mémoire d'une Odeur",
  "Gucci Memoire d'une Odeur": "Mémoire d'une Odeur",
  "Mémoire d'une Odeur": "Mémoire d'une Odeur",
  
  // === ARIANA GRANDE ===
  "Ari": "Ari",
  "Cloud": "Cloud",
  
  // === MÁS MARCAS ===
  "Acqua di Gio": "Acqua di Gio",
  "Giorgio Armani Acqua di Gio": "Acqua di Gio",
  "Acqua di Gio Profumo": "Acqua di Gio Profumo",
  "Acqua di Gio Profondo": "Acqua di Gio Profondo",
  
  "Terre d'Hermès": "Terre d'Hermès",
  "Hermès Terre d'Hermès": "Terre d'Hermès",
  
  "Paco Rabanne 1 Million": "1 Million",
  "1 Million": "1 Million",
  
  "Le Labo Santal 33": "Santal 33",
  "Santal 33": "Santal 33",
  
  "Prada L'Homme": "L'Homme",
  
  // === MARCAS CON VARIACIONES ===
  "Maison Francis Kurkdjian Baccarat Rouge 540": "Baccarat Rouge 540",
  "Baccarat Rouge 540": "Baccarat Rouge 540",
  
  "Maison Margiela REPLICA Jazz Club": "Jazz Club",
  "Jazz Club": "Jazz Club",
  
  "Lancôme La Vie Est Belle": "La Vie Est Belle",
  "La Vie Est Belle": "La Vie Est Belle",
  
  "Valentino Voce Viva": "Voce Viva",
  "Voce Viva": "Voce Viva",
  
  "Hugo Boss Boss Bottled": "Boss Bottled",
  "Boss Bottled": "Boss Bottled",
  
  // === PERFUMES DE CELEBRIDADES ===
  "Fenty Eau de Parfum": "Fenty",
  "Fenty": "Fenty",
  
  "Kilian Love, don't be shy": "Love, don't be shy",
  "Love, don't be shy": "Love, don't be shy",
  
  "Glow by JLo": "Glow",
  
  "Lady Gaga Fame": "Fame",
  "Fame": "Fame",
  
  "S by Shakira": "S",
  
  "Billie Eilish Eilish": "Eilish",
  
  "Selena Gomez Rare": "Rare",
  
  "KKW Crystal Gardenia": "Crystal Gardenia",
  
  "Omnia": "Omnia",
};

// Función para normalizar nombres de perfumes
export const normalizePerfumeName = (name) => {
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
  
  // 3. Eliminar nombres de marcas comunes al principio
  const cleanName = trimmedName
    .replace(/^(Creed|Tom Ford|Chanel|Dior|YSL|Yves Saint Laurent|Jo Malone|Gucci|Armani|Hermès|Prada|Le Labo|Maison Margiela|Lancôme|Valentino|Hugo Boss|Kilian|Ariana Grande|Lady Gaga|Selena Gomez|Billie Eilish|Shakira|Jennifer Lopez|Fenty|KKW)\s+/i, '')
    .trim();
  
  // Si después de limpiar queda algo, usarlo
  if (cleanName && cleanName !== trimmedName) {
    return cleanName;
  }
  
  // 4. Devolver el nombre original
  return trimmedName;
};

// Función para obtener variaciones comunes de un nombre de perfume
export const getPerfumeVariations = (name) => {
  if (!name) return [''];
  
  const variations = [name];
  
  // Añadir versión sin marca
  const withoutBrand = name.replace(/^(Creed|Tom Ford|Chanel|Dior|YSL|Yves Saint Laurent|Jo Malone|Gucci)\s+/i, '').trim();
  if (withoutBrand && withoutBrand !== name) {
    variations.push(withoutBrand);
  }
  
  // Añadir versión en inglés si es diferente
  const englishName = name
    .replace('Mémoire', 'Memoire')
    .replace('N°5', 'No. 5')
    .replace('J\'adore', 'Jadore')
    .replace('Ombré', 'Ombre');
  
  if (englishName !== name) {
    variations.push(englishName);
  }
  
  return variations;
};

export default perfumeNameMapping;
const fs = require('fs');
const path = require('path');

// --- 1. BANCO DE DATOS (BASES PARA MEZCLAR) ---

const femaleCelebs = [
  "Rihanna", "Taylor Swift", "Beyoncé", "Adele", "Ariana Grande", "Billie Eilish", "Dua Lipa", 
  "Selena Gomez", "Katy Perry", "Jennifer Lopez", "Angelina Jolie", "Margot Robbie", "Scarlett Johansson", 
  "Zendaya", "Emma Stone", "Jennifer Aniston", "Julia Roberts", "Nicole Kidman", "Blake Lively", 
  "Penélope Cruz", "Marilyn Monroe", "Audrey Hepburn", "Lady Gaga", "Madonna", "Shakira", 
  "Karol G", "Rosalía", "Miley Cyrus", "Demi Lovato", "Hailey Bieber", "Kendall Jenner", 
  "Kylie Jenner", "Kim Kardashian", "Gigi Hadid", "Bella Hadid", "Gal Gadot", "Ana de Armas",
  "Natalie Portman", "Emma Watson", "Anne Hathaway", "Sandra Bullock", "Meryl Streep", "Charlize Theron",
  "Salma Hayek", "Sofía Vergara", "Thalía", "Paulina Rubio", "Nathy Peluso", "Aitana", "Lola Índigo"
];

const maleCelebs = [
  "Brad Pitt", "Johnny Depp", "Leonardo DiCaprio", "George Clooney", "Tom Cruise", "Robert Pattinson", 
  "Chris Hemsworth", "Ryan Reynolds", "The Rock", "Will Smith", "Frank Sinatra", "Michael Jackson", 
  "David Beckham", "Cristiano Ronaldo", "Lionel Messi", "Harry Styles", "Justin Bieber", "Drake", 
  "Jay-Z", "Kanye West", "Bad Bunny", "Maluma", "J Balvin", "Daddy Yankee", "Rauw Alejandro",
  "Cillian Murphy", "Pedro Pascal", "Henry Cavill", "Tom Holland", "Timothée Chalamet", "Keanu Reeves",
  "Robert Downey Jr.", "Chris Evans", "Mark Zuckerberg", "Elon Musk", "Jeff Bezos", "Lewis Hamilton",
  "Fernando Alonso", "Rafael Nadal", "Roger Federer", "LeBron James", "Travis Scott", "The Weeknd",
  "Bruno Mars", "Ed Sheeran", "Eminem", "Snoop Dogg", "50 Cent", "Vin Diesel", "Jason Statham"
];

const perfumesWomen = [
  { name: "Love, Don't Be Shy", brand: "Kilian", img: "https://fimgs.net/mdimg/perfume/375x500.2886.jpg" },
  { name: "Santal Blush", brand: "Tom Ford", img: "https://fimgs.net/mdimg/perfume/375x500.12765.jpg" },
  { name: "Flowerbomb", brand: "Viktor&Rolf", img: "https://fimgs.net/mdimg/perfume/375x500.1460.jpg" },
  { name: "Libre", brand: "YSL", img: "https://fimgs.net/mdimg/perfume/375x500.56077.jpg" },
  { name: "Baccarat Rouge 540", brand: "Maison Francis Kurkdjian", img: "https://fimgs.net/mdimg/perfume/375x500.33519.jpg" },
  { name: "Black Orchid", brand: "Tom Ford", img: "https://fimgs.net/mdimg/perfume/375x500.1018.jpg" },
  { name: "Delina", brand: "Parfums de Marly", img: "https://fimgs.net/mdimg/perfume/375x500.43871.jpg" },
  { name: "Good Girl", brand: "Carolina Herrera", img: "https://fimgs.net/mdimg/perfume/375x500.39681.jpg" },
  { name: "La Vie Est Belle", brand: "Lancôme", img: "https://fimgs.net/mdimg/perfume/375x500.14982.jpg" },
  { name: "Chanel No 5", brand: "Chanel", img: "https://fimgs.net/mdimg/perfume/375x500.608.jpg" },
  { name: "Miss Dior", brand: "Dior", img: "https://fimgs.net/mdimg/perfume/375x500.193.jpg" },
  { name: "Daisy", brand: "Marc Jacobs", img: "https://fimgs.net/mdimg/perfume/375x500.1361.jpg" },
  { name: "Lost Cherry", brand: "Tom Ford", img: "https://fimgs.net/mdimg/perfume/375x500.51258.jpg" },
  { name: "Alien", brand: "Mugler", img: "https://fimgs.net/mdimg/perfume/375x500.707.jpg" },
  { name: "Opium", brand: "YSL", img: "https://fimgs.net/mdimg/perfume/375x500.7399.jpg" }
];

const perfumesMen = [
  { name: "Sauvage", brand: "Dior", img: "https://fimgs.net/mdimg/perfume/375x500.31861.jpg" },
  { name: "Aventus", brand: "Creed", img: "https://fimgs.net/mdimg/perfume/375x500.9828.jpg" },
  { name: "Bleu de Chanel", brand: "Chanel", img: "https://fimgs.net/mdimg/perfume/375x500.12408.jpg" },
  { name: "Acqua di Gio", brand: "Giorgio Armani", img: "https://fimgs.net/mdimg/perfume/375x500.410.jpg" },
  { name: "One Million", brand: "Paco Rabanne", img: "https://fimgs.net/mdimg/perfume/375x500.3747.jpg" },
  { name: "Terre d'Hermes", brand: "Hermès", img: "https://fimgs.net/mdimg/perfume/375x500.17.jpg" },
  { name: "Oud Wood", brand: "Tom Ford", img: "https://fimgs.net/mdimg/perfume/375x500.1826.jpg" },
  { name: "Tobacco Vanille", brand: "Tom Ford", img: "https://fimgs.net/mdimg/perfume/375x500.1825.jpg" },
  { name: "Le Male", brand: "Jean Paul Gaultier", img: "https://fimgs.net/mdimg/perfume/375x500.430.jpg" },
  { name: "Eros", brand: "Versace", img: "https://fimgs.net/mdimg/perfume/375x500.16670.jpg" },
  { name: "Boss Bottled", brand: "Hugo Boss", img: "https://fimgs.net/mdimg/perfume/375x500.383.jpg" },
  { name: "The One for Men", brand: "Dolce&Gabbana", img: "https://fimgs.net/mdimg/perfume/375x500.1722.jpg" },
  { name: "Santal 33", brand: "Le Labo", img: "https://fimgs.net/mdimg/perfume/375x500.12201.jpg" },
  { name: "Layton", brand: "Parfums de Marly", img: "https://fimgs.net/mdimg/perfume/375x500.39314.jpg" },
  { name: "Jazz Club", brand: "Maison Margiela", img: "https://fimgs.net/mdimg/perfume/375x500.20541.jpg" }
];

// --- 2. FUNCIÓN GENERADORA ---

function generateDatabase() {
  console.log('🏭 Iniciando fábrica de datos...');
  const database = [];
  let idCounter = 1;

  // Función auxiliar para obtener random de un array
  const getRandom = (arr) => arr[Math.floor(Math.random() * arr.length)];
  
  // Función para generar perfumes aleatorios (1 o 2 por persona)
  const getPerfumes = (sourceArray) => {
    const num = Math.random() > 0.7 ? 2 : 1; // 30% de chance de tener 2 perfumes
    const selected = [];
    for(let i=0; i<num; i++) {
      const p = getRandom(sourceArray);
      if(!selected.find(x => x.name === p.name)) selected.push(p);
    }
    return selected;
  };

  // Generar mujeres
  femaleCelebs.forEach(name => {
    database.push({
      id: idCounter++,
      celebrity: name,
      gender: "Femenino",
      perfumes: getPerfumes(perfumesWomen)
    });
  });

  // Generar hombres
  maleCelebs.forEach(name => {
    database.push({
      id: idCounter++,
      celebrity: name,
      gender: "Masculino",
      perfumes: getPerfumes(perfumesMen)
    });
  });

  // Generar 800 entradas adicionales "Random" para probar rendimiento
  // Usaremos nombres genéricos o variaciones
  for (let i = 0; i < 800; i++) {
    const isFemale = Math.random() > 0.5;
    const baseName = isFemale ? getRandom(femaleCelebs) : getRandom(maleCelebs);
    const gender = isFemale ? "Femenino" : "Masculino";
    const perfumeList = isFemale ? perfumesWomen : perfumesMen;
    
    // Crear variación del nombre para que no sean idénticos
    // Ej: "Brad Pitt (Fan)", "Brad Pitt II", etc. 
    // O simulamos usuarios famosos de instagram/tiktok
    const suffix = Math.floor(Math.random() * 9999);
    
    database.push({
      id: idCounter++,
      celebrity: `${baseName} User${suffix}`, // Simulación de influencers
      gender: gender,
      perfumes: getPerfumes(perfumeList)
    });
  }

  // --- 3. ESCRIBIR ARCHIVO ---
  
  const fileContent = `// BASE DE DATOS GENERADA AUTOMÁTICAMENTE
// Fecha: ${new Date().toLocaleString()}
// Cantidad: ${database.length} registros

export const CELEBRITIES_DB = ${JSON.stringify(database, null, 2)};
`;

  const outputPath = path.join(__dirname, '..', 'src', 'data', 'celebritiesDB.js');
  
  // Asegurar que el directorio existe
  const dir = path.dirname(outputPath);
  if (!fs.existsSync(dir)){
      fs.mkdirSync(dir, { recursive: true });
  }

  fs.writeFileSync(outputPath, fileContent);
  console.log(`✅ Éxito! Base de datos generada en: ${outputPath}`);
  console.log(`📊 Total de famosos: ${database.length}`);
}

// Ejecutar
generateDatabase();
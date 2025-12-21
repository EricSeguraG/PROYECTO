

const celebritiesDB = [
  {
    "id": 1,
    "celebrity": "Rihanna",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Fenty Eau de Parfum",
        "brand": "Fenty",
        
      },
      {
        "name": "Kilian Love, don't be shy",
        "brand": "Kilian",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46534.jpg"
      }
    ]
  },
  {
    "id": 2,
    "celebrity": "Brad Pitt",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Tobacco Vanille",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27445.jpg"
      }
    ]
  },
  {
    "id": 3,
    "celebrity": "Cristiano Ronaldo",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Baccarat Rouge 540",
        "brand": "Maison Francis Kurkdjian",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46535.jpg"
      }
    ]
  },
  {
    "id": 4,
    "celebrity": "Harry Styles",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Gucci Mémoire d'une Odeur",
        "brand": "Gucci",
        "img": "https://fimgs.net/mdimg/perfume/375x500.59366.jpg"
      },
      {
        "name": "Tom Ford Noir Extreme",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46536.jpg"
      }
    ]
  },
  {
    "id": 5,
    "celebrity": "Taylor Swift",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Tom Ford Velvet Orchid",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46537.jpg"
      },
      {
        "name": "Jo Malone Wood Sage & Sea Salt",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46538.jpg"
      }
    ]
  },
  {
    "id": 6,
    "celebrity": "David Beckham",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Green Irish Tweed",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15877.jpg"
      },
      {
        "name": "Acqua di Gio",
        "brand": "Giorgio Armani",
        "img": "https://fimgs.net/mdimg/perfume/375x500.6480.jpg"
      }
    ]
  },
  {
    "id": 7,
    "celebrity": "Beyoncé",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Tom Ford Black Orchid",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15878.jpg"
      },
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      }
    ]
  },
  {
    "id": 8,
    "celebrity": "Johnny Depp",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      },
      {
        "name": "Creed Bois du Portugal",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15880.jpg"
      }
    ]
  },
  {
    "id": 9,
    "celebrity": "Zendaya",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Valentino Voce Viva",
        "brand": "Valentino",
        "img": "https://fimgs.net/mdimg/perfume/375x500.67488.jpg"
      },
      {
        "name": "Lancôme La Vie Est Belle",
        "brand": "Lancôme",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46539.jpg"
      }
    ]
  },
  {
    "id": 10,
    "celebrity": "Ryan Reynolds",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Silver Mountain Water",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15881.jpg"
      },
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      }
    ]
  },
  {
    "id": 11,
    "celebrity": "Jennifer Lopez",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Glow by JLo",
        "brand": "Jennifer Lopez",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68033.jpg"
      },
      {
        "name": "Tom Ford Santal Blush",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46540.jpg"
      }
    ]
  },
  {
    "id": 12,
    "celebrity": "Kanye West",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Maison Margiela REPLICA Jazz Club",
        "brand": "Maison Margiela",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46541.jpg"
      }
    ]
  },
  {
    "id": 13,
    "celebrity": "Lady Gaga",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Fame",
        "brand": "Lady Gaga",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68034.jpg"
      },
      {
        "name": "Black Opium",
        "brand": "YSL",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46542.jpg"
      }
    ]
  },
  {
    "id": 14,
    "celebrity": "Robert Downey Jr.",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Millésime Impérial",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15882.jpg"
      },
      {
        "name": "Dior Homme Intense",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46543.jpg"
      }
    ]
  },
  {
    "id": 15,
    "celebrity": "Ariana Grande",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Ari",
        "brand": "Ariana Grande",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68035.jpg"
      },
      {
        "name": "Cloud",
        "brand": "Ariana Grande",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68036.jpg"
      }
    ]
  },
  {
    "id": 16,
    "celebrity": "George Clooney",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Original Santal",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15883.jpg"
      },
      {
        "name": "Terre d'Hermès",
        "brand": "Hermès",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15884.jpg"
      }
    ]
  },
  {
    "id": 17,
    "celebrity": "Blake Lively",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Chance Eau Tendre",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46544.jpg"
      },
      {
        "name": "Jo Malone English Pear & Freesia",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46545.jpg"
      }
    ]
  },
  {
    "id": 18,
    "celebrity": "LeBron James",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Viking",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68038.jpg"
      },
      {
        "name": "Boss Bottled",
        "brand": "Hugo Boss",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46546.jpg"
      }
    ]
  },
  {
    "id": 19,
    "celebrity": "Emma Watson",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Jo Malone Lime Basil & Mandarin",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46547.jpg"
      },
      {
        "name": "Chanel N°5 L'Eau",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68039.jpg"
      }
    ]
  },
  {
    "id": 20,
    "celebrity": "Will Smith",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Oud Wood",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46548.jpg"
      }
    ]
  },
  {
    "id": 21,
    "celebrity": "Kim Kardashian",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "KKW Crystal Gardenia",
        "brand": "KKW Fragrance",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68040.jpg"
      },
      {
        "name": "Tom Ford Lost Cherry",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68041.jpg"
      }
    ]
  },
  {
    "id": 22,
    "celebrity": "Drake",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Baccarat Rouge 540",
        "brand": "Maison Francis Kurkdjian",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46535.jpg"
      }
    ]
  },
  {
    "id": 23,
    "celebrity": "Margot Robbie",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Jo Malone Peony & Blush Suede",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68042.jpg"
      }
    ]
  },
  {
    "id": 24,
    "celebrity": "Tom Cruise",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Dior Fahrenheit",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68043.jpg"
      }
    ]
  },
  {
    "id": 25,
    "celebrity": "Dua Lipa",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "YSL Libre",
        "brand": "Yves Saint Laurent",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68044.jpg"
      },
      {
        "name": "Maison Francis Kurkdjian Baccarat Rouge 540",
        "brand": "Maison Francis Kurkdjian",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46535.jpg"
      }
    ]
  },
  {
    "id": 26,
    "celebrity": "Leonardo DiCaprio",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Green Irish Tweed",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15877.jpg"
      },
      {
        "name": "Acqua di Gio Profumo",
        "brand": "Giorgio Armani",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68045.jpg"
      }
    ]
  },
  {
    "id": 27,
    "celebrity": "Billie Eilish",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Eilish",
        "brand": "Billie Eilish",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68046.jpg"
      },
      {
        "name": "Tom Ford Black Orchid",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15878.jpg"
      }
    ]
  },
  {
    "id": 28,
    "celebrity": "Chris Hemsworth",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 29,
    "celebrity": "Selena Gomez",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Rare",
        "brand": "Selena Gomez",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68047.jpg"
      },
      {
        "name": "Cloud",
        "brand": "Ariana Grande",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68036.jpg"
      }
    ]
  },
  {
    "id": 30,
    "celebrity": "The Weeknd",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Ombré Leather",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68048.jpg"
      }
    ]
  },
  {
    "id": 31,
    "celebrity": "Gal Gadot",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Miss Dior",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.6478.jpg"
      }
    ]
  },
  {
    "id": 32,
    "celebrity": "Tom Holland",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Homme Sport",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68049.jpg"
      }
    ]
  },
  {
    "id": 33,
    "celebrity": "Cardi B",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Libre",
        "brand": "YSL",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68044.jpg"
      },
      {
        "name": "Good Girl",
        "brand": "Carolina Herrera",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46519.jpg"
      }
    ]
  },
  {
    "id": 34,
    "celebrity": "Michael B. Jordan",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Tobacco Vanille",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27445.jpg"
      }
    ]
  },
  {
    "id": 35,
    "celebrity": "Zoe Kravitz",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Tom Ford Black Orchid",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15878.jpg"
      },
      {
        "name": "Le Labo Santal 33",
        "brand": "Le Labo",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68050.jpg"
      }
    ]
  },
  {
    "id": 36,
    "celebrity": "Timothée Chalamet",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 37,
    "celebrity": "Lupita Nyong'o",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Jo Malone Wood Sage & Sea Salt",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46538.jpg"
      }
    ]
  },
  {
    "id": 38,
    "celebrity": "Chris Evans",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 39,
    "celebrity": "Doja Cat",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Good Girl",
        "brand": "Carolina Herrera",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46519.jpg"
      },
      {
        "name": "Black Opium",
        "brand": "YSL",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46542.jpg"
      }
    ]
  },
  {
    "id": 40,
    "celebrity": "Jason Momoa",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 41,
    "celebrity": "Anya Taylor-Joy",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Jo Malone English Pear & Freesia",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46545.jpg"
      }
    ]
  },
  {
    "id": 42,
    "celebrity": "Pedro Pascal",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 43,
    "celebrity": "Florence Pugh",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Chance Eau Tendre",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46544.jpg"
      },
      {
        "name": "Jo Malone Wood Sage & Sea Salt",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46538.jpg"
      }
    ]
  },
  {
    "id": 44,
    "celebrity": "Oscar Isaac",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Oud Wood",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46548.jpg"
      }
    ]
  },
  {
    "id": 45,
    "celebrity": "Jenna Ortega",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Libre",
        "brand": "YSL",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68044.jpg"
      },
      {
        "name": "Cloud",
        "brand": "Ariana Grande",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68036.jpg"
      }
    ]
  },
  {
    "id": 46,
    "celebrity": "Austin Butler",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 47,
    "celebrity": "Sydney Sweeney",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Good Girl",
        "brand": "Carolina Herrera",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46519.jpg"
      },
      {
        "name": "Black Opium",
        "brand": "YSL",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46542.jpg"
      }
    ]
  },
  {
    "id": 48,
    "celebrity": "Jacob Elordi",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 49,
    "celebrity": "Rachel Zegler",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Jo Malone English Pear & Freesia",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46545.jpg"
      }
    ]
  },
  {
    "id": 50,
    "celebrity": "Paul Mescal",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 51,
    "celebrity": "Megan Thee Stallion",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Good Girl",
        "brand": "Carolina Herrera",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46519.jpg"
      },
      {
        "name": "Libre",
        "brand": "YSL",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68044.jpg"
      }
    ]
  },
  {
    "id": 52,
    "celebrity": "Jonathan Majors",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Tobacco Vanille",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27445.jpg"
      }
    ]
  },
  {
    "id": 53,
    "celebrity": "Jodie Comer",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Jo Malone Wood Sage & Sea Salt",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46538.jpg"
      }
    ]
  },
  {
    "id": 54,
    "celebrity": "Barry Keoghan",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 55,
    "celebrity": "Ayo Edebiri",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Chance Eau Tendre",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46544.jpg"
      },
      {
        "name": "Jo Malone English Pear & Freesia",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46545.jpg"
      }
    ]
  },
  {
    "id": 56,
    "celebrity": "Charles Melton",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 57,
    "celebrity": "Greta Gerwig",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Jo Malone Wood Sage & Sea Salt",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46538.jpg"
      }
    ]
  },
  {
    "id": 58,
    "celebrity": "Brett Goldstein",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 59,
    "celebrity": "Da'Vine Joy Randolph",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Libre",
        "brand": "YSL",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68044.jpg"
      },
      {
        "name": "Good Girl",
        "brand": "Carolina Herrera",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46519.jpg"
      }
    ]
  },
  {
    "id": 60,
    "celebrity": "Colman Domingo",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Oud Wood",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46548.jpg"
      }
    ]
  },
  {
    "id": 61,
    "celebrity": "Lily-Rose Depp",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Jo Malone Wood Sage & Sea Salt",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46538.jpg"
      }
    ]
  },
  {
    "id": 62,
    "celebrity": "Josh O'Connor",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 63,
    "celebrity": "Maya Hawke",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Chance Eau Tendre",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46544.jpg"
      },
      {
        "name": "Jo Malone English Pear & Freesia",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46545.jpg"
      }
    ]
  },
  {
    "id": 64,
    "celebrity": "Paul Giamatti",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Tobacco Vanille",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27445.jpg"
      }
    ]
  },
  {
    "id": 65,
    "celebrity": "Julianne Moore",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Jo Malone Wood Sage & Sea Salt",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46538.jpg"
      }
    ]
  },
  {
    "id": 66,
    "celebrity": "Jeffrey Wright",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 67,
    "celebrity": "America Ferrera",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Libre",
        "brand": "YSL",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68044.jpg"
      },
      {
        "name": "Good Girl",
        "brand": "Carolina Herrera",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46519.jpg"
      }
    ]
  },
  {
    "id": 68,
    "celebrity": "Sterling K. Brown",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Oud Wood",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46548.jpg"
      }
    ]
  },
  {
    "id": 69,
    "celebrity": "Danielle Brooks",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Jo Malone English Pear & Freesia",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46545.jpg"
      }
    ]
  },
  {
    "id": 70,
    "celebrity": "Mark Ruffalo",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 71,
    "celebrity": "Carey Mulligan",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Chance Eau Tendre",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46544.jpg"
      },
      {
        "name": "Jo Malone Wood Sage & Sea Salt",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46538.jpg"
      }
    ]
  },
  {
    "id": 72,
    "celebrity": "Robert De Niro",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Tobacco Vanille",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27445.jpg"
      }
    ]
  },
  {
    "id": 73,
    "celebrity": "Jodie Foster",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Jo Malone English Pear & Freesia",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46545.jpg"
      }
    ]
  },
  {
    "id": 74,
    "celebrity": "John Krasinski",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 75,
    "celebrity": "Emily Blunt",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Libre",
        "brand": "YSL",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68044.jpg"
      },
      {
        "name": "Good Girl",
        "brand": "Carolina Herrera",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46519.jpg"
      }
    ]
  },
  {
    "id": 76,
    "celebrity": "Ryan Gosling",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Oud Wood",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46548.jpg"
      }
    ]
  },
  {
    "id": 77,
    "celebrity": "Margot Robbie",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Jo Malone Wood Sage & Sea Salt",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46538.jpg"
      }
    ]
  },
  {
    "id": 78,
    "celebrity": "Cillian Murphy",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 79,
    "celebrity": "Florence Pugh",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Chance Eau Tendre",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46544.jpg"
      },
      {
        "name": "Jo Malone English Pear & Freesia",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46545.jpg"
      }
    ]
  },
  {
    "id": 80,
    "celebrity": "Matt Damon",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Tobacco Vanille",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27445.jpg"
      }
    ]
  },
  {
    "id": 81,
    "celebrity": "Natalie Portman",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Jo Malone Wood Sage & Sea Salt",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46538.jpg"
      }
    ]
  },
  {
    "id": 82,
    "celebrity": "Joaquin Phoenix",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 83,
    "celebrity": "Viola Davis",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Libre",
        "brand": "YSL",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68044.jpg"
      },
      {
        "name": "Good Girl",
        "brand": "Carolina Herrera",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46519.jpg"
      }
    ]
  },
  {
    "id": 84,
    "celebrity": "Adam Driver",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Oud Wood",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46548.jpg"
      }
    ]
  },
  {
    "id": 85,
    "celebrity": "Regina King",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Jo Malone English Pear & Freesia",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46545.jpg"
      }
    ]
  },
  {
    "id": 86,
    "celebrity": "Mahershala Ali",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 87,
    "celebrity": "Lupita Nyong'o",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Chance Eau Tendre",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46544.jpg"
      },
      {
        "name": "Jo Malone Wood Sage & Sea Salt",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46538.jpg"
      }
    ]
  },
  {
    "id": 88,
    "celebrity": "Riz Ahmed",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Tobacco Vanille",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27445.jpg"
      }
    ]
  },
  {
    "id": 89,
    "celebrity": "Yara Shahidi",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Jo Malone Wood Sage & Sea Salt",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46538.jpg"
      }
    ]
  },
  {
    "id": 90,
    "celebrity": "Lakeith Stanfield",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 91,
    "celebrity": "Zoe Saldana",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Libre",
        "brand": "YSL",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68044.jpg"
      },
      {
        "name": "Good Girl",
        "brand": "Carolina Herrera",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46519.jpg"
      }
    ]
  },
  {
    "id": 92,
    "celebrity": "Donald Glover",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Oud Wood",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46548.jpg"
      }
    ]
  },
  {
    "id": 93,
    "celebrity": "Issa Rae",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Jo Malone English Pear & Freesia",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46545.jpg"
      }
    ]
  },
  {
    "id": 94,
    "celebrity": "John Boyega",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 95,
    "celebrity": "Keke Palmer",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Chance Eau Tendre",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46544.jpg"
      },
      {
        "name": "Jo Malone Wood Sage & Sea Salt",
        "brand": "Jo Malone",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46538.jpg"
      }
    ]
  },
  {
    "id": 96,
    "celebrity": "Daniel Kaluuya",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Tobacco Vanille",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27445.jpg"
      }
    ]
  },
  {
    "id": 97,
    "celebrity": "Marina Moreno",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Omnia",
        "brand": "Bvlgari",
      "img": "videos/51k20QOF3+L._AC_SX679_.jpg"

      }
    ]
  },
  {
    "id": 98,
    "celebrity": "Winston Duke",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 99,
    "celebrity": "Letitia Wright",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Libre",
        "brand": "YSL",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68044.jpg"
      },
      {
        "name": "Good Girl",
        "brand": "Carolina Herrera",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46519.jpg"
      }
    ]
  },
  {
    "id": 100,
    "celebrity": "Jonathan Groff",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Oud Wood",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46548.jpg"
      }
    ]
  },
  {
    "id": 101,
    "celebrity": "Messi",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      }
    ]
  },
  {
    "id": 102,
    "celebrity": "Bad Bunny",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Tom Ford Ombré Leather",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68048.jpg"
      },
      {
        "name": "Maison Francis Kurkdjian Baccarat Rouge 540",
        "brand": "Maison Francis Kurkdjian",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46535.jpg"
      }
    ]
  },
  {
    "id": 103,
    "celebrity": "Anuel AA",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Paco Rabanne 1 Million",
        "brand": "Paco Rabanne",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27420.jpg"
      },
      {
        "name": "Dior Sauvage Elixir",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68051.jpg"
      }
    ]
  },
  {
    "id": 104,
    "celebrity": "Karol G",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Good Girl",
        "brand": "Carolina Herrera",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46519.jpg"
      },
      {
        "name": "Libre",
        "brand": "YSL",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68044.jpg"
      },
      {
        "name": "Black Opium",
        "brand": "YSL",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46542.jpg"
      }
    ]
  },
  {
    "id": 105,
    "celebrity": "Shakira",
    "gender": "Femenino",
    "perfumes": [
      {
        "name": "Chanel Coco Mademoiselle",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.15879.jpg"
      },
      {
        "name": "Dior J'adore",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.6479.jpg"
      },
      {
        "name": "S by Shakira",
        "brand": "Shakira",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68052.jpg"
      }
    ]
  },
  {
    "id": 106,
    "celebrity": "Rauw Alejandro",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Tom Ford Noir Extreme",
        "brand": "Tom Ford",
        "img": "https://fimgs.net/mdimg/perfume/375x500.46536.jpg"
      }
    ]
  },
  {
    "id": 107,
    "celebrity": "Vinicius Junior",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Creed Aventus",
        "brand": "Creed",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27433.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      },
      {
        "name": "Acqua di Gio Profondo",
        "brand": "Giorgio Armani",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68053.jpg"
      }
    ]
  },
  {
    "id": 108,
    "celebrity": "Lamine Yamal",
    "gender": "Masculino",
    "perfumes": [
      {
        "name": "Bleu de Chanel",
        "brand": "Chanel",
        "img": "https://fimgs.net/mdimg/perfume/375x500.27419.jpg"
      },
      {
        "name": "Dior Sauvage",
        "brand": "Dior",
        "img": "https://fimgs.net/mdimg/perfume/375x500.42212.jpg"
      },
      {
        "name": "Prada L'Homme",
        "brand": "Prada",
        "img": "https://fimgs.net/mdimg/perfume/375x500.68054.jpg"
      }
    ]
  },


];


export default celebritiesDB;
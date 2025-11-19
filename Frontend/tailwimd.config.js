// tailwind.config.js
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
};
module.exports = {
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          500: '#0ea5a4', // teal-ish
          700: '#0f172a'  // deep indigo
        },
        accent: {
          500: '#ff6b6b' // coral
        }
      },
      borderRadius: { '2xl': '1rem' }
    }
  }
}

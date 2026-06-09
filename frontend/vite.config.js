import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the FastAPI backend on :8000, so the frontend
// can use relative URLs and we sidestep CORS entirely in development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.PORT) || 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});

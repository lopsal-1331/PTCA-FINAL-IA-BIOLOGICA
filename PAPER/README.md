# uc3mmaster — plantilla LaTeX (Máster en Ciencia y Tecnología, estilo UC3M)

- `uc3mmaster.cls`: clase. Opciones: `memoria`, `deber`, `apuntes`; y `doscolumnas` (combinable, p. ej. `\documentclass[apuntes,doscolumnas]{uc3mmaster}`; `unacolumna` es el valor por defecto). En dos columnas usa `table*`/`figure*` para elementos a ancho completo.
- `ejemplo.tex`: documento de ejemplo con portada, contraportada y entornos.
- Compilar: `latexmk -pdf ejemplo.tex` (pdfLaTeX). En Overleaf: subir el zip, compilador pdfLaTeX.
- Colores en la clase (`uc3mazul`, `uc3mcian`, ...): ajústalos a tu gusto.
- Entornos: `definicion`, `teorema`, `proposicion`, `ejemplo`, `nota`, `resultado`, `pseudocodigo` (líneas con `\pl`, sangría `\ind`, palabras clave `\kw{}`).
- Metadatos: `\titulo \subtitulo \asignatura \tipo \curso \fecha \autor{Nombre}{NIA}`; `\logo{fichero}` opcional.
- No se incluye el logotipo oficial de la UC3M (marca registrada): añade el tuyo con `\logo`.
- `ejemplo-dos-columnas.tex`: ejemplo con la opción `doscolumnas`.

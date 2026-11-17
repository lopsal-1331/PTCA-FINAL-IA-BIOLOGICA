%% uc3mmaster.cls -- Plantilla para memorias, deberes y apuntes
%% Máster en Ciencia y Tecnología (estilo inspirado en la identidad de la UC3M)
%% Compila con pdfLaTeX (Overleaf: compilador pdfLaTeX, TeX Live reciente).
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{uc3mmaster}[2026/10/25 Plantilla UC3M Master]

% ---------- Opciones ----------
\newif\ifuc@apuntes \uc@apuntesfalse
\newif\ifuc@deber   \uc@deberfalse
\DeclareOption{memoria}{\uc@apuntesfalse\uc@deberfalse}
\DeclareOption{deber}{\uc@debertrue}
\DeclareOption{apuntes}{\uc@apuntestrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[a4paper,10.5pt]{article}

% ---------- Paquetes ----------
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage[spanish,es-tabla,es-nodecimaldot]{babel}
\RequirePackage{mathpazo}
\RequirePackage[scaled=0.95]{helvet}
\RequirePackage{microtype}
\RequirePackage{amsmath,amssymb,mathtools}
\RequirePackage{xcolor}
\RequirePackage[a4paper,top=2.6cm,bottom=2.4cm,left=2.4cm,right=2.4cm,headheight=15pt]{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{enumitem}
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{graphicx}
\RequirePackage{caption}
\RequirePackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds}
\RequirePackage[most]{tcolorbox}
\RequirePackage{lastpage}
\RequirePackage[colorlinks=true,linkcolor=uc3mazul,citecolor=uc3mazul,urlcolor=uc3mcian!70!black]{hyperref}

% ---------- Colores (ajustables) ----------
\definecolor{uc3mazul}{HTML}{003A70}   % azul oscuro corporativo
\definecolor{uc3mcian}{HTML}{00A6D6}   % acento claro
\definecolor{uc3mgris}{HTML}{5A6772}
\definecolor{uc3mfondo}{HTML}{EEF3F8}

% ---------- Metadatos ----------
\def\uc@titulo{Título del trabajo}
\def\uc@subtitulo{}
\def\uc@asignatura{Asignatura}
\def\uc@programa{Máster en Ciencia y Tecnología}
\def\uc@universidad{Universidad Carlos III de Madrid}
\def\uc@escuela{Escuela Politécnica Superior}
\def\uc@curso{Curso 2026--2027}
\def\uc@tipo{\ifuc@apuntes Apuntes\else\ifuc@deber Práctica\else Memoria\fi\fi}
\def\uc@fecha{\today}
\def\uc@autores{}
\newcommand{\titulo}[1]{\def\uc@titulo{#1}}
\newcommand{\subtitulo}[1]{\def\uc@subtitulo{#1}}
\newcommand{\asignatura}[1]{\def\uc@asignatura{#1}}
\newcommand{\programa}[1]{\def\uc@programa{#1}}
\newcommand{\curso}[1]{\def\uc@curso{#1}}
\newcommand{\tipo}[1]{\def\uc@tipo{#1}}
\newcommand{\fecha}[1]{\def\uc@fecha{#1}}
\newcommand{\logo}[1]{\def\uc@logo{#1}}   % opcional: ruta a un logo propio
% \autor{Nombre Apellidos}{NIA o correo}  (repetir por cada integrante)
\newcommand{\autor}[2]{\g@addto@macro\uc@autores{\item[] \textbf{#1}\ifx&#2&\else\hfill{\small\color{uc3mgris}#2}\fi}}

% ---------- Tipografía y secciones ----------
\renewcommand{\familydefault}{\rmdefault}
\newcommand{\uc@sf}{\sffamily\bfseries\color{uc3mazul}}
\titleformat{\section}{\uc@sf\Large}{\thesection}{0.7em}{}[{\color{uc3mcian}\titlerule[1.2pt]}]
\titleformat{\subsection}{\uc@sf\large}{\thesubsection}{0.7em}{}
\titleformat{\subsubsection}{\uc@sf\normalsize}{\thesubsubsection}{0.7em}{}
\titlespacing*{\section}{0pt}{1.3ex plus .4ex}{0.9ex}
\titlespacing*{\subsection}{0pt}{1.1ex plus .3ex}{0.6ex}
\setlist{nosep,leftmargin=1.6em,itemsep=2pt}
\setlist[itemize]{label=\textcolor{uc3mcian}{\textbullet}}
\setlist[enumerate]{label=\textcolor{uc3mazul}{\arabic*.}}
\captionsetup{font={small},labelfont={bf,color=uc3mazul},labelsep=period}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt plus 1pt}

% ---------- Cabecera y pie ----------
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.6pt}
\renewcommand{\footrulewidth}{0pt}
\fancyhead[L]{\sffamily\small\color{uc3mgris}\uc@asignatura}
\fancyhead[R]{\sffamily\small\color{uc3mgris}\uc@titulo}
\fancyfoot[L]{\sffamily\scriptsize\color{uc3mgris}\uc@programa\ -- UC3M}
\fancyfoot[R]{\sffamily\small\color{uc3mazul}\thepage\,/\,\pageref{LastPage}}
\renewcommand{\headrule}{\color{uc3mcian}\hrule width\headwidth height\headrulewidth}
\fancypagestyle{plain}{\fancyhf{}\renewcommand{\headrulewidth}{0pt}%
  \fancyfoot[R]{\sffamily\small\color{uc3mazul}\thepage\,/\,\pageref{LastPage}}}

% ---------- Portada ----------
\newcommand{\makeportada}{%
\begin{titlepage}
\thispagestyle{empty}
\begin{tikzpicture}[remember picture,overlay]
  \fill[uc3mazul] (current page.north west) rectangle ([yshift=-9.6cm]current page.north east);
  \fill[uc3mcian] ([yshift=-9.6cm]current page.north west) rectangle ([yshift=-9.85cm]current page.north east);
  \fill[uc3mazul] (current page.south west) rectangle ([yshift=1.3cm]current page.south east);
  \fill[uc3mcian] ([yshift=1.3cm]current page.south west) rectangle ([yshift=1.45cm]current page.south east);
\end{tikzpicture}
\vspace*{1.2cm}
{\color{white}\sffamily
 \ifx\uc@logo\undefined
   {\Large\bfseries\uc@universidad}\\[2pt]{\small\uc@escuela}%
 \else
   \includegraphics[height=1.6cm]{\uc@logo}%
 \fi
 \par\vspace{1.1cm}
 {\small\bfseries\MakeUppercase{\uc@tipo}\ \textbullet\ \uc@asignatura}\par\vspace{6pt}
 {\fontsize{20}{30}\selectfont\bfseries\uc@titulo\par}
 \ifx\uc@subtitulo\empty\else\vspace{6pt}{\large\uc@subtitulo\par}\fi}
\vspace{2.6cm}
{\sffamily\color{uc3mazul}
 \begin{itemize}[label={},leftmargin=0pt,itemsep=5pt]
   \uc@autores
 \end{itemize}
 \vspace{0.6cm}
 {\color{uc3mgris}\small \uc@programa\\ \uc@curso\\ \uc@fecha}}
\vfill
{\color{white}\sffamily\small\uc@universidad\ \textbullet\ \uc@escuela}
\vspace{0.1cm}
\end{titlepage}
\setcounter{page}{2}}

% ---------- Contraportada ----------
\newcommand{\makecontraportada}{%
\clearpage
\thispagestyle{empty}
\begin{tikzpicture}[remember picture,overlay]
  \fill[uc3mazul] (current page.north west) rectangle (current page.south east);
  \fill[uc3mcian] ([yshift=5cm]current page.south west) rectangle ([yshift=5.2cm]current page.south east);
\end{tikzpicture}
\vspace*{\fill}
\begin{center}\color{white}\sffamily
 {\Large\bfseries\uc@universidad}\\[4pt]
 {\uc@escuela}\\[2pt]
 {\uc@programa}\\[14pt]
 {\small\uc@asignatura\ \textbullet\ \uc@curso}
\end{center}
\vspace*{3cm}}

% ---------- Entornos con caja ----------
\tcbset{uc3mbox/.style={enhanced,breakable,colback=uc3mfondo,colframe=uc3mazul,
  boxrule=0pt,leftrule=2.5pt,arc=0pt,left=6pt,right=6pt,top=3pt,bottom=3pt,
  fonttitle=\sffamily\bfseries\small,coltitle=uc3mazul,colbacktitle=uc3mfondo,
  attach boxed title to top left={yshift=-2mm},boxed title style={boxrule=0pt,frame hidden},
  before skip=6pt,after skip=6pt}}
\newtcolorbox{definicion}[1][]{uc3mbox,title={Definición\ifx&#1&\else\ (#1)\fi}}
\newtcolorbox{teorema}[1][]{uc3mbox,title={Teorema\ifx&#1&\else\ (#1)\fi}}
\newtcolorbox{proposicion}[1][]{uc3mbox,title={Proposición\ifx&#1&\else\ (#1)\fi}}
\newtcolorbox{ejemplo}[1][]{uc3mbox,colframe=uc3mcian,coltitle=uc3mcian!60!black,title={Ejemplo\ifx&#1&\else\ (#1)\fi}}
\newtcolorbox{nota}[1][]{uc3mbox,colback=white,colframe=uc3mgris,coltitle=uc3mgris,colbacktitle=white,title={Nota\ifx&#1&\else\ (#1)\fi}}
\newtcolorbox{resultado}[1][]{uc3mbox,colback=uc3mazul!6,title={#1}}

% Pseudocódigo sin dependencias externas
\newcounter{uc@pl}
\newtcolorbox[auto counter]{pseudocodigo}[1]{uc3mbox,colback=white,
  title={Algoritmo \thetcbcounter: #1},before upper={\setcounter{uc@pl}{0}\small}}
\newcommand{\pl}{\par\stepcounter{uc@pl}\makebox[1.6em][r]{\footnotesize\color{uc3mgris}\theuc@pl}\hspace{0.7em}}
\newcommand{\ind}{\hspace{1.3em}}
\newcommand{\kw}[1]{\textbf{#1}}

% ---------- Atajos matemáticos generales ----------
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\argmax}{arg\,max}
\newcommand{\N}{\mathbb{N}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\set}[1]{\left\{#1\right\}}
\newcommand{\abs}[1]{\left|#1\right|}
\newcommand{\defeq}{\coloneqq}
\AtBeginDocument{\hypersetup{pdftitle={\uc@titulo},pdfauthor={UC3M Master}}}
\endinput

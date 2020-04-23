\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{diss}[2019/12/23]

\LoadClass[11pt, twoside, openright]{book}
\RequirePackage{graphicx}

\RequirePackage[hidelinks]{hyperref}
\RequirePackage{subcaption}
\RequirePackage{siunitx}
\RequirePackage{censor}
\RequirePackage{enumerate}
\RequirePackage{mathtools}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{xfrac}
\linespread{1.2}
\RequirePackage{amsthm}
\RequirePackage{greekletters}

\RequirePackage{tikz-qtree}

\RequirePackage{tikz}
\RequirePackage{tkz-graph}
\usetikzlibrary{arrows}
\usetikzlibrary{shadings}
\usetikzlibrary{arrows.meta}
\RequirePackage[utf8]{inputenc}
\RequirePackage[a4paper, bindingoffset=1cm, top=2.5cm, right=2cm, left=2cm, bottom=3cm, footskip=1.5cm]{geometry} 
\RequirePackage{fancyhdr}

\RequirePackage{listings}
\lstset{
extendedchars=\true,
inputencoding=utf8x
}
\RequirePackage[export]{adjustbox}
\RequirePackage{longtable}
\RequirePackage{supertabular}   % Page breaks within tables
\RequirePackage{multicol}       % Allows table cells to span cols
\RequirePackage{multirow}       % Allows table cells to span rows
\RequirePackage{hhline}         % Horizontal lines in tables

\RequirePackage{floatrow}

\RequirePackage[ruled, lined, linesnumbered, noend]{algorithm2e}

\RequirePackage{lipsum}

\graphicspath{ {../src/figures/} }

\RequirePackage{array}
\newcolumntype{L}[1]{>{\raggedright\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\newcolumntype{R}[1]{>{\raggedleft\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}

\RequirePackage{xcolor}

\definecolor{codegreen}{rgb}{0,0.65,0.4}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.6,0.25,0.5}
\definecolor{backcolour}{rgb}{0.95,0.95,0.95}
\definecolor{codeblue}{rgb}{0.5,0.5,0.95}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},
    commentstyle=\itshape,
%    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
%   stringstyle=\color{codepurple},
%    basicstyle=\footnotesize,
    breakatwhitespace=false,
%    breaklines=true,
%    captionpos=b,
%    keepspaces=true,
    numbers=right,
%    numbersep=5pt,
    showspaces=false,
%    showstringspaces=false,
%    showtabs=false,
    tabsize=5,
    basicstyle = \ttfamily\footnotesize,
%    identifierstyle = \color{codepurple},
%
}



\RequirePackage{tocloft}
\tocloftpagestyle{fancychapter}

\fancypagestyle{fancypage}{
	\fancyhf{}
	\fancyfoot[LE,RO]{\thepage}
	\fancyhead[LE]{\leftmark}
	\fancyhead[RO]{\rightmark}
	\renewcommand{\headrulewidth}{0pt}
	\renewcommand{\footrulewidth}{0pt}
}

\fancypagestyle{fancychapter}{
	\fancyhf{}
	\fancyfoot[LE,RO]{\thepage}
	\fancyhead[LE]{}
	\fancyhead[RO]{}
	\renewcommand{\headrulewidth}{0pt}
	\renewcommand{\footrulewidth}{0pt}
}


\pagestyle{fancypage}

\renewcommand{\maketitle}{%
	\thispagestyle{empty}%
		\begin{flushright} \textbf{\author}\\
		\textit{\crsid}\end{flushright}\ \\\ \\\ \\\ \\\ \\\ \\
	\begin{center}%
		{\LARGE \title}\\\ \\\ \\\ \\
		\textit{A Dissertation In} \\\ \\
		{\Large Part II \textit{of the} Computer Science Tripos}\\\ \\
		\textit{Submitted In}\\\ \\
		{\Large \year}\\\ \\	
		\textit{By A Member Of}\\\ \\
		{\Large \college}
	\end{center}%
	\clearpage\thispagestyle{empty}\ \clearpage
}

\let\origdoublepage\cleardoublepage
\newcommand{\clearemptydoublepage}{%
	\clearpage
	{\pagestyle{empty}\origdoublepage}%
}
\let\cleardoublepage\clearemptydoublepage

\theoremstyle{definition}
\makeatletter
\def\namedlabel#1#2{\begingroup
    #2%
    \def\@currentlabel{#2}%
    \phantomsection\label{#1}\endgroup
}
\renewcommand\chapter{\cleardoublepage%
			\thispagestyle{fancychapter}% original style: plain
			\global\@topnum\z@
			\@afterindentfalse
			\secdef\@chapter\@schapter}
\def\@endpart{}
\def\@endchapter{}

\newtheorem*{rep@theorem}{\rep@title}
\newcommand{\newreptheorem}[2]{%
\newenvironment{rep#1}[1]{%
 \def\rep@title{#2 \ref{##1}}%
 \begin{rep@theorem}}%
 {\end{rep@theorem}}}
\makeatother

\newtheorem{definition}{Definition}[chapter]

\newtheorem{theorem}{Theorem}[section]
\newtheorem{corollary}{Corollary}[theorem]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem*{claim}{Claim}
\newreptheorem{theorem}{Theorem}
\newreptheorem{lemma}{Lemma}

\theoremstyle{remark}
\newtheorem*{comment}{Comment}

\DeclareOption{unit}{\renewcommand{\coursetype}{Unit of Assessment }}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax

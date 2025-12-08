" ============================================================
"   ABHI'S NEON MATRIX VIM CONFIG (NO PLUGINS)
"   Pure .vimrc. Pure violence. Pure drip.
" ============================================================

" ----- BASIC SETTINGS --------------------------------------------------------

set nocompatible
syntax on
filetype plugin indent on

set number
set relativenumber
set cursorline
set termguicolors
set nowrap
set showcmd
set noshowmode
set laststatus=2
set showtabline=2

set tabstop=4
set shiftwidth=4
set expandtab
set autoindent
set smartindent

set ignorecase
set smartcase

set mouse=a
set hidden
set clipboard=unnamedplus

set updatetime=200
set timeoutlen=400

set noswapfile
set nobackup
set undofile

set wildmenu
set wildmode=longest:full,full

" Make backspace behave like a normal person
set backspace=indent,eol,start

" ----- MATRIX-NEON COLORSCHEME (BUILT-IN) ------------------------------------

hi clear
if exists("syntax_on")
  syntax reset
endif
let g:colors_name = "neon_matrix"

let s:bg      = "#050811"
let s:bg_d    = "#02040a"
let s:bg_l    = "#101320"
let s:fg      = "#c8d3f5"
let s:fg_dim  = "#7a88c7"
let s:cyan    = "#00f0ff"
let s:cyan_d  = "#00b3c4"
let s:mag     = "#ff007c"
let s:green   = "#a3ff12"
let s:yellow  = "#ffc857"
let s:red     = "#ff4b5c"
let s:orange  = "#ff9f1c"

" Core UI
exe "hi Normal       guifg=" . s:fg     . " guibg=" . s:bg
exe "hi CursorLine   guibg=" . s:bg_l
exe "hi CursorLineNr guifg=" . s:cyan   . " guibg=" . s:bg_l . " gui=bold"
exe "hi LineNr       guifg=#4b5263 guibg=" . s:bg
exe "hi Visual       guibg=#24314a"
exe "hi VertSplit    guifg=#15182a guibg=" . s:bg
exe "hi ColorColumn  guibg=#15182a"
exe "hi Pmenu        guifg=" . s:fg     . " guibg=" . s:bg_l
exe "hi PmenuSel     guifg=" . s:bg     . " guibg=" . s:cyan
exe "hi StatusLine   guifg=" . s:fg     . " guibg=" . s:bg_l
exe "hi StatusLineNC guifg=" . s:fg_dim . " guibg=" . s:bg_d
exe "hi TabLine      guifg=" . s:fg_dim . " guibg=" . s:bg_d
exe "hi TabLineSel   guifg=" . s:cyan   . " guibg=" . s:bg_l . " gui=bold"
exe "hi TabLineFill  guifg=" . s:fg_dim . " guibg=" . s:bg_d

" Syntax
exe "hi Comment      guifg=#4b5263 gui=italic"
exe "hi Constant     guifg=" . s:yellow
exe "hi String       guifg=" . s:green
exe "hi Function     guifg=" . s:cyan . " gui=bold"
exe "hi Identifier   guifg=" . s:cyan
exe "hi Statement    guifg=" . s:mag
exe "hi Keyword      guifg=" . s:mag . " gui=italic"
exe "hi PreProc      guifg=" . s:orange
exe "hi Type         guifg=" . s:cyan_d
exe "hi Error        guifg=" . s:bg    . " guibg=" . s:red . " gui=bold"
exe "hi WarningMsg   guifg=" . s:orange . " gui=bold"

" Diffs
exe "hi DiffAdd      guifg=" . s:green  . " guibg=#0b2b19"
exe "hi DiffChange   guifg=" . s:yellow . " guibg=#26210b"
exe "hi DiffDelete   guifg=" . s:red    . " guibg=#2b0b12"

" ----- CURSOR CGI MODE ----------------------------------------
set guicursor=n-v-c:block,i-ci:ver25,r-cr:hor20,o:hor50

" ----- STATUSLINE (SEXY AS HELL) -------------------------------
function! NeonMode()
  let l:m = mode()
  return l:m ==# 'n' ? 'NORMAL' :
       \ l:m ==# 'i' ? 'INSERT' :
       \ l:m ==# 'v' ? 'VISUAL' :
       \ l:m ==# 'V' ? 'V-LINE' :
       \ l:m ==# '' ? 'V-BLOCK' :
       \ l:m ==# 'R' ? 'REPLACE' :
       \ l:m ==# 'c' ? 'COMMAND' :
       \ l:m
endfunction

set statusline=
set statusline+=%#StatusLine#\ ⦿\ 
set statusline+=%f\ 
set statusline+=%m%r%h%w
set statusline+=%=
set statusline+=%#StatusLine#\ [%{&filetype!=''?&filetype:'no-ft'}]\ 
set statusline+=%#StatusLine#\ %{NeonMode()}\ 
set statusline+=%#StatusLine#\ ln\ %l/%L\ col\ %c\ 

" ----- TABLINE ------------------------------------------------
function! NeonTabline()
  let s = ""
  let tcount = tabpagenr('$')
  for i in range(1, tcount)
    let buflist = tabpagebuflist(i)
    let winnr = tabpagewinnr(i)
    let name = fnamemodify(bufname(buflist[winnr - 1]), ":t")
    if empty(name)
      let name = "[No Name]"
    endif
    if i == tabpagenr()
      let s .= "%#TabLineSel# " . i . ": " . name . " "
    else
      let s .= "%#TabLine# " . i . ": " . name . " "
    endif
  endfor
  let s .= "%#TabLineFill#"
  return s
endfunction
set tabline=%!NeonTabline()

" ----- KEYMAPS -------------------------------------------------

let mapleader=" "

" Better splits navigation
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

" Splits
nnoremap <leader>v :vsplit<CR>
nnoremap <leader>h :split<CR>

" Save/Quit
nnoremap <leader>w :w<CR>
nnoremap <leader>q :q<CR>
nnoremap <leader>Q :qa!<CR>

" Clear search
nnoremap <leader><space> :nohlsearch<CR>

" Fast escape
inoremap jk <Esc>

" Move lines up/down
vnoremap J :m '>+1<CR>gv=gv
vnoremap K :m '<-2<CR>gv=gv

" Terminal
nnoremap <leader>tt :split | terminal<CR>

" Toggle relativenumber
nnoremap <leader>rn :set relativenumber!<CR>

" ----- FILETYPE-SPECIFIC SETTINGS -----------------------------

augroup AbhiLang
  autocmd!

  " Python = 4 spaces
  autocmd FileType python  setlocal tabstop=4 shiftwidth=4 expandtab

  " Bash/Zsh = 2 spaces
  autocmd FileType sh,bash,zsh setlocal tabstop=2 shiftwidth=2 expandtab

  " C/C++ = 4 spaces
  autocmd FileType c,cpp,h,hpp setlocal tabstop=4 shiftwidth=4 expandtab

  " Rust = 4 spaces
  autocmd FileType rust setlocal tabstop=4 shiftwidth=4 expandtab

  " Go = tabs, width 4
  autocmd FileType go setlocal noexpandtab tabstop=4 shiftwidth=4
augroup END

" ----- DIM INACTIVE WINDOWS -----------------------------------
augroup NeonDim
  autocmd!
  autocmd WinEnter,BufWinEnter * setlocal cursorline
  autocmd WinLeave * setlocal nocursorline
augroup END

" ----- BASIC SETTINGS -----------------------------------------

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

set backspace=indent,eol,start

" ----- BUILT-IN NEON COLORSCHEME ------------------------------

hi clear
if exists("syntax_on")
  syntax reset
endif
let g:colors_name = "neon_matrix"

" Core UI
hi Normal       guifg=#c8d3f5 guibg=#050811
hi SignColumn   guifg=#7a88c7 guibg=#050811
hi CursorLine   guibg=#101320
hi CursorLineNr guifg=#00f0ff guibg=#101320 gui=bold
hi LineNr       guifg=#4b5263 guibg=#050811
hi Visual       guibg=#24314a
hi VertSplit    guifg=#15182a guibg=#050811
hi ColorColumn  guibg=#15182a

hi Pmenu        guifg=#c8d3f5 guibg=#101320
hi PmenuSel     guifg=#050811 guibg=#00f0ff
hi PmenuSbar    guibg=#02040a
hi PmenuThumb   guibg=#00b3c4

hi StatusLine   guifg=#c8d3f5 guibg=#101320
hi StatusLineNC guifg=#7a88c7 guibg=#02040a
hi TabLine      guifg=#7a88c7 guibg=#02040a
hi TabLineSel   guifg=#00f0ff guibg=#101320 gui=bold
hi TabLineFill  guifg=#7a88c7 guibg=#02040a

" Syntax
hi Comment      guifg=#4b5263 gui=italic
hi Constant     guifg=#ffc857
hi String       guifg=#a3ff12
hi Character    guifg=#a3ff12
hi Number       guifg=#ffc857
hi Boolean      guifg=#ffc857
hi Identifier   guifg=#00f0ff
hi Function     guifg=#00f0ff gui=bold
hi Statement    guifg=#ff007c
hi Conditional  guifg=#ff007c
hi Repeat       guifg=#ff007c
hi Operator     guifg=#c8d3f5
hi Keyword      guifg=#ff007c gui=italic
hi Exception    guifg=#ff007c
hi PreProc      guifg=#ff9f1c
hi Include      guifg=#ff9f1c
hi Define       guifg=#ff9f1c
hi Type         guifg=#00b3c4
hi StorageClass guifg=#00b3c4
hi Structure    guifg=#00b3c4
hi Typedef      guifg=#00b3c4
hi Special      guifg=#ffc857
hi Delimiter    guifg=#c8d3f5
hi Error        guifg=#050811 guibg=#ff4b5c gui=bold
hi ErrorMsg     guifg=#050811 guibg=#ff4b5c gui=bold
hi WarningMsg   guifg=#ff9f1c gui=bold
hi MoreMsg      guifg=#a3ff12 gui=bold
hi Question     guifg=#a3ff12 gui=bold

" Diffs
hi DiffAdd      guifg=#a3ff12 guibg=#0b2b19
hi DiffChange   guifg=#ffc857 guibg=#26210b
hi DiffDelete   guifg=#ff4b5c guibg=#2b0b12
hi DiffText     guifg=#00f0ff guibg=#0b2230

" ----- CURSOR STYLE (SAFE) ------------------------------------

if exists('&guicursor')
  set guicursor=n-v-c:block,i-ci:ver25,r-cr:hor20,o:hor50
endif

" ----- STATUSLINE ---------------------------------------------

function! NeonMode()
  let l:m = mode()
  if l:m ==# 'n'  | return 'NORMAL'
  elseif l:m ==# 'i' | return 'INSERT'
  elseif l:m ==# 'v' | return 'VISUAL'
  elseif l:m ==# 'V' | return 'V-LINE'
  elseif l:m ==# "" | return 'V-BLOCK'
  elseif l:m ==# 'R' | return 'REPLACE'
  elseif l:m ==# 'c' | return 'COMMAND'
  endif
  return l:m
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
  let s = ''
  let tcount = tabpagenr('$')
  for i in range(1, tcount)
    let buflist = tabpagebuflist(i)
    let winnr = tabpagewinnr(i)
    if len(buflist) == 0
      let name = '[No Name]'
    else
      let bnr = buflist[winnr - 1]
      let name = fnamemodify(bufname(bnr), ':t')
      if empty(name)
        let name = '[No Name]'
      endif
    endif
    if i == tabpagenr()
      let s .= '%#TabLineSel# ' . i . ': ' . name . ' '
    else
      let s .= '%#TabLine# ' . i . ': ' . name . ' '
    endif
  endfor
  let s .= '%#TabLineFill#'
  return s
endfunction

set tabline=%!NeonTabline()

" ----- KEYMAPS ------------------------------------------------

let mapleader=" "

" Window nav
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

" Splits
nnoremap <leader>v :vsplit<CR>
nnoremap <leader>h :split<CR>

" Save / quit
nnoremap <leader>w :w<CR>
nnoremap <leader>q :q<CR>
nnoremap <leader>Q :qa!<CR>

" Clear search
nnoremap <leader><space> :nohlsearch<CR>

" Fast escape
inoremap jk <Esc>

" Move selected lines
vnoremap J :m '>+1<CR>gv=gv
vnoremap K :m '<-2<CR>gv=gv

" Terminal
nnoremap <leader>tt :split | terminal<CR>

" Toggle relativenumber
nnoremap <leader>rn :set relativenumber!<CR>

" ----- FILETYPE SETTINGS --------------------------------------

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

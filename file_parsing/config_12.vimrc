set nocompatible
filetype plugin indent on
syntax on
set t_Co=256
set number
set ruler
set cursorline
set laststatus=2
set showmatch
set incsearch
set hlsearch
set wildmenu
set autoindent
set smartindent
set expandtab
set tabstop=4
set shiftwidth=4
set softtabstop=4
set backspace=indent,eol,start
highlight Normal ctermbg=NONE ctermfg=253
highlight NonText ctermbg=NONE ctermfg=240
highlight Comment ctermfg=242
highlight Constant ctermfg=141
highlight String ctermfg=48
highlight Character ctermfg=141
highlight Number ctermfg=141
highlight Boolean ctermfg=141
highlight Float ctermfg=141
highlight Identifier ctermfg=81
highlight Function ctermfg=118
highlight Statement ctermfg=197
highlight Conditional ctermfg=197
highlight Repeat ctermfg=197
highlight Label ctermfg=197
highlight Operator ctermfg=197
highlight Keyword ctermfg=197
highlight Exception ctermfg=197
highlight PreProc ctermfg=208
highlight Include ctermfg=208
highlight Define ctermfg=208
highlight Macro ctermfg=208
highlight PreCondit ctermfg=208
highlight Type ctermfg=81
highlight StorageClass ctermfg=81
highlight Structure ctermfg=81
highlight Typedef ctermfg=81
highlight Special ctermfg=208
highlight SpecialChar ctermfg=208
highlight Tag ctermfg=208
highlight Delimiter ctermfg=253
highlight SpecialComment ctermfg=208
highlight Debug ctermfg=208
highlight Underlined ctermfg=81 cterm=underline
highlight Ignore ctermfg=240
highlight Error ctermfg=196 ctermbg=NONE
highlight Todo ctermfg=196 ctermbg=NONE
highlight LineNr ctermfg=240 ctermbg=NONE
highlight CursorLine ctermbg=236 cterm=NONE
highlight CursorLineNr ctermfg=253 ctermbg=236
highlight StatusLine ctermfg=232 ctermbg=118
highlight StatusLineNC ctermfg=232 ctermbg=244
highlight Search ctermfg=232 ctermbg=220
highlight Visual ctermfg=232 ctermbg=118
highlight MatchParen ctermfg=232 ctermbg=208
autocmd FileType python setlocal expandtab shiftwidth=4 tabstop=4 softtabstop=4
autocmd FileType sh setlocal expandtab shiftwidth=4 tabstop=4 softtabstop=4
autocmd FileType make setlocal noexpandtab

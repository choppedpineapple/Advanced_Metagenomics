" Abhi Night — a low-contrast, beautiful Vim theme
" Author: Dark
" Feel: Soft dark, slightly desaturated, calming tones

set background=dark
highlight clear
if exists("syntax_on")
  syntax reset
endif
let g:colors_name = "dark_night"

" Base palette
let s:bg       = "#1c1c1c"
let s:bg_dark  = "#161616"
let s:fg       = "#d0d0d0"
let s:fg_dim   = "#999999"
let s:accent   = "#89b482"   " subtle green
let s:blue     = "#7aa2f7"
let s:purple   = "#bb9af7"
let s:orange   = "#e0af68"
let s:red      = "#f7768e"
let s:cyan     = "#7dcfff"
let s:comment  = "#707070"
let s:cursor   = "#c0caf5"

" General
hi Normal        guifg=s:fg       guibg=s:bg
hi CursorLine    guibg=#202020
hi CursorColumn  guibg=#202020
hi LineNr        guifg=#555555    guibg=s:bg_dark
hi CursorLineNr  guifg=s:orange   guibg=#202020 gui=bold
hi Comment       guifg=s:comment  gui=italic
hi Constant      guifg=s:purple
hi String        guifg=s:accent
hi Character     guifg=s:accent
hi Number        guifg=s:orange
hi Boolean       guifg=s:orange
hi Identifier    guifg=s:cyan
hi Function      guifg=s:blue     gui=bold
hi Statement     guifg=s:purple   gui=bold
hi Keyword       guifg=s:purple
hi Conditional   guifg=s:red
hi Repeat        guifg=s:red
hi Operator      guifg=s:fg
hi Exception     guifg=s:red      gui=italic
hi Type          guifg=s:accent   gui=bold
hi StorageClass  guifg=s:cyan
hi Structure     guifg=s:accent
hi Typedef       guifg=s:cyan
hi Special       guifg=s:orange
hi PreProc       guifg=s:blue
hi Include       guifg=s:blue
hi Define        guifg=s:orange
hi Macro         guifg=s:cyan
hi PreCondit     guifg=s:orange
hi Title         guifg=s:blue     gui=bold
hi Todo          guifg=s:red      guibg=#2a2a2a gui=bold,italic
hi Visual        guibg=#333333
hi Search        guibg=#404040    guifg=s:orange
hi IncSearch     guibg=s:orange   guifg=s:bg
hi VertSplit     guifg=#333333
hi StatusLine    guibg=#2a2a2a    guifg=s:fg gui=bold
hi StatusLineNC  guibg=#1e1e1e    guifg=s:fg_dim
hi Pmenu         guibg=#2a2a2a    guifg=s:fg
hi PmenuSel      guibg=s:blue     guifg=s:bg
hi PmenuSbar     guibg=#2a2a2a
hi PmenuThumb    guibg=#555555
hi DiffAdd       guibg=#203020    guifg=s:accent
hi DiffChange    guibg=#303020
hi DiffDelete    guibg=#301a1a    guifg=s:red
hi DiffText      guibg=#403030    guifg=s:orange
hi Underlined    guifg=s:cyan     gui=underline
hi MatchParen    guibg=#303030    guifg=s:orange gui=bold
hi Directory     guifg=s:blue
hi Error         guibg=#3a1a1a    guifg=s:red gui=bold
hi WarningMsg    guifg=s:orange
hi ErrorMsg      guifg=s:red      guibg=#2a1a1a
hi SignColumn    guibg=s:bg_dark  guifg=s:fg_dim

" Subtle folds & background tone
hi Folded        guibg=#202020    guifg=s:fg_dim gui=italic
hi FoldColumn    guibg=s:bg_dark  guifg=s:fg_dim
hi NonText       guifg=#333333
hi EndOfBuffer   guifg=#333333
hi Conceal       guifg=s:fg_dim   guibg=s:bg

" Cursor
hi Cursor        guifg=s:bg       guibg=s:cursor
hi TermCursor    guifg=s:bg       guibg=s:cursor

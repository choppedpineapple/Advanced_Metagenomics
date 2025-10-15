if exists("g:colors_name")
  finish
endif

set background=dark
if has("termguicolors")
  set termguicolors
endif

let g:colors_name = "dark_night"

" Palette (literal hex values)
" bg:        #1c1c1c
" bg_dark:   #161616
" fg:        #d0d0d0
" fg_dim:    #999999
" accent:    #89b482
" blue:      #7aa2f7
" purple:    #bb9af7
" orange:    #e0af68
" red:       #f7768e
" cyan:      #7dcfff
" comment:   #707070
" cursor:    #c0caf5

hi clear
syntax reset

hi Normal        guifg=#d0d0d0    guibg=#1c1c1c
hi CursorLine    guibg=#202020
hi CursorColumn  guibg=#202020
hi LineNr        guifg=#555555    guibg=#161616
hi CursorLineNr  guifg=#e0af68    guibg=#202020 gui=bold
hi Comment       guifg=#707070    gui=italic
hi Constant      guifg=#bb9af7
hi String        guifg=#89b482
hi Character     guifg=#89b482
hi Number        guifg=#e0af68
hi Boolean       guifg=#e0af68
hi Identifier    guifg=#7dcfff
hi Function      guifg=#7aa2f7    gui=bold
hi Statement     guifg=#bb9af7    gui=bold
hi Keyword       guifg=#bb9af7
hi Conditional   guifg=#f7768e
hi Repeat        guifg=#f7768e
hi Operator      guifg=#d0d0d0
hi Exception     guifg=#f7768e    gui=italic
hi Type          guifg=#89b482    gui=bold
hi StorageClass  guifg=#7dcfff
hi Structure     guifg=#89b482
hi Typedef       guifg=#7dcfff
hi Special       guifg=#e0af68
hi PreProc       guifg=#7aa2f7
hi Include       guifg=#7aa2f7
hi Define        guifg=#e0af68
hi Macro         guifg=#7dcfff
hi PreCondit     guifg=#e0af68
hi Title         guifg=#7aa2f7    gui=bold
hi Todo          guifg=#f7768e    guibg=#2a2a2a gui=bold,italic
hi Visual        guibg=#333333
hi Search        guibg=#404040    guifg=#e0af68
hi IncSearch     guibg=#e0af68    guifg=#1c1c1c
hi VertSplit     guifg=#333333
hi StatusLine    guibg=#2a2a2a    guifg=#d0d0d0 gui=bold
hi StatusLineNC  guibg=#1e1e1e    guifg=#999999
hi Pmenu         guibg=#2a2a2a    guifg=#d0d0d0
hi PmenuSel      guibg=#7aa2f7    guifg=#1c1c1c
hi PmenuSbar     guibg=#2a2a2a
hi PmenuThumb    guibg=#555555
hi DiffAdd       guibg=#203020    guifg=#89b482
hi DiffChange    guibg=#303020
hi DiffDelete    guibg=#301a1a    guifg=#f7768e
hi DiffText      guibg=#403030    guifg=#e0af68
hi Underlined    guifg=#7dcfff    gui=underline
hi MatchParen    guibg=#303030    guifg=#e0af68 gui=bold
hi Directory     guifg=#7aa2f7
hi Error         guibg=#3a1a1a    guifg=#f7768e gui=bold
hi WarningMsg    guifg=#e0af68
hi ErrorMsg      guifg=#f7768e    guibg=#2a1a1a
hi SignColumn    guibg=#161616    guifg=#999999

hi Folded        guibg=#202020    guifg=#999999 gui=italic
hi FoldColumn    guibg=#161616    guifg=#999999
hi NonText       guifg=#333333
hi EndOfBuffer   guifg=#333333
hi Conceal       guifg=#999999    guibg=#1c1c1c

hi Cursor        guifg=#1c1c1c     guibg=#c0caf5
hi TermCursor    guifg=#1c1c1c     guibg=#c0caf5

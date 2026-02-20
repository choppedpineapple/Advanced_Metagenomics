-- =========================================================
-- Basic Editor Settings
-- =========================================================
local opt = vim.opt

-- Line numbering
opt.number = true
opt.relativenumber = true

-- Tab and indentation settings (4 spaces)
opt.tabstop = 4
opt.shiftwidth = 4
opt.expandtab = true
opt.autoindent = true
opt.smartindent = true

-- Search settings
opt.ignorecase = true
opt.smartcase = true
opt.hlsearch = false
opt.incsearch = true

-- UI and behavior settings
opt.termguicolors = true
opt.cursorline = true
opt.wrap = false
opt.signcolumn = "yes"
opt.scrolloff = 8
opt.updatetime = 50

-- =========================================================
-- Custom Dim Purplish-Blue Colorscheme
-- =========================================================
-- Clear any existing background and syntax highlighting
vim.cmd([[
  highlight clear
  if exists("syntax_on")
    syntax reset
  endif
]])

vim.o.background = "dark"
vim.g.colors_name = "dim_purple_custom"

-- Helper function to set highlight groups
local function set_hl(group, options)
    vim.api.nvim_set_hl(0, group, options)
end

-- Color Palette
local bg = "#1a1823"
local fg = "#b4b0c5"
local comment = "#5b5770"
local keyword = "#8c7bb8"
local string_col = "#6b8eab"
local func = "#7a95c4"
local type_col = "#9082b0"
local constant = "#9c7e9e"
local selection = "#2d2a40"
local cursorline_bg = "#211e2e"

-- Base UI Highlights
set_hl("Normal", { fg = fg, bg = bg })
set_hl("NormalFloat", { fg = fg, bg = cursorline_bg })
set_hl("CursorLine", { bg = cursorline_bg })
set_hl("CursorLineNr", { fg = keyword, bold = true })
set_hl("LineNr", { fg = comment })
set_hl("Visual", { bg = selection })
set_hl("Search", { bg = keyword, fg = bg })
set_hl("SignColumn", { bg = bg })
set_hl("ColorColumn", { bg = cursorline_bg })
set_hl("EndOfBuffer", { fg = bg })

-- Syntax Highlights
set_hl("Comment", { fg = comment, italic = true })
set_hl("Keyword", { fg = keyword })
set_hl("Conditional", { fg = keyword })
set_hl("Repeat", { fg = keyword })
set_hl("String", { fg = string_col })
set_hl("Function", { fg = func })
set_hl("Identifier", { fg = fg })
set_hl("Type", { fg = type_col })
set_hl("Constant", { fg = constant })
set_hl("Number", { fg = constant })
set_hl("Operator", { fg = fg })
set_hl("Statement", { fg = keyword })
set_hl("PreProc", { fg = keyword })

-- =========================================================
-- Language Specific Overrides
-- =========================================================
local autocmd = vim.api.nvim_create_autocmd
local augroup = vim.api.nvim_create_augroup("LangOverrides", { clear = true })

-- Python, C, C++, and Rust
autocmd("FileType", {
    group = augroup,
    pattern = { "python", "c", "cpp", "rust" },
    callback = function()
        vim.opt_local.tabstop = 4
        vim.opt_local.shiftwidth = 4
        vim.opt_local.expandtab = true
    end,
})

-- Bash
autocmd("FileType", {
    group = augroup,
    pattern = "sh",
    callback = function()
        vim.opt_local.tabstop = 4
        vim.opt_local.shiftwidth = 4
        vim.opt_local.expandtab = true
    end,
})

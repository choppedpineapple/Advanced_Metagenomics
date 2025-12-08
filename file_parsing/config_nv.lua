-----------------------------------------------------------
--  Single-File, No-Plugin, Neon Neovim Setup
--  Drop this as ~/.config/nvim/init.lua
-----------------------------------------------------------

-----------------------------------------------------------
-- 1. Leader & Basic Globals
-----------------------------------------------------------
vim.g.mapleader = " "
vim.g.maplocalleader = " "

-----------------------------------------------------------
-- 2. Core Options (UI, Editing, Search)
-----------------------------------------------------------
local opt = vim.opt

-- UI
opt.number = true
opt.relativenumber = true
opt.cursorline = true
opt.termguicolors = true
opt.signcolumn = "yes"
opt.wrap = false
opt.scrolloff = 6
opt.sidescrolloff = 8
opt.showmode = false        -- we’ll show mode in statusline instead
opt.laststatus = 3          -- global statusline
opt.showtabline = 2         -- always show tabline

-- Editing
opt.expandtab = true
opt.shiftwidth = 4
opt.tabstop = 4
opt.smartindent = true
opt.autoindent = true
opt.smartcase = true
opt.ignorecase = true
opt.incsearch = true
opt.hlsearch = true
opt.clipboard = "unnamedplus"
opt.hidden = true           -- keep buffers in background
opt.updatetime = 200        -- faster CursorHold, diagnostic updates
opt.timeoutlen = 400        -- faster mapped sequence timeout

-- Files & backups
opt.swapfile = false
opt.backup = false
opt.undofile = true

-- Command-line completion
opt.wildmenu = true
opt.wildmode = { "longest:full", "full" }

-- Enable syntax & filetype detection
vim.cmd("syntax on")
vim.cmd("filetype plugin indent on")

-----------------------------------------------------------
-- 3. Custom Neon Colorscheme (No external colorscheme)
-----------------------------------------------------------
local function set_neon_theme()
  -- Clear existing highlights
  vim.cmd("highlight clear")
  if vim.fn.exists("syntax_on") == 1 then
    vim.cmd("syntax reset")
  end
  vim.o.background = "dark"
  vim.g.colors_name = "neon_matrix"

  local hl = vim.api.nvim_set_hl

  -- Base palette
  local bg      = "#050811"
  local bg_d    = "#02040a"
  local bg_l    = "#101320"
  local fg      = "#c8d3f5"
  local fg_dim  = "#7a88c7"
  local cyan    = "#00f0ff"
  local cyan_d  = "#00b3c4"
  local magenta = "#ff007c"
  local green   = "#a3ff12"
  local yellow  = "#ffc857"
  local red     = "#ff4b5c"
  local orange  = "#ff9f1c"

  -- Core UI
  hl(0, "Normal",        { fg = fg,      bg = bg })
  hl(0, "NormalFloat",   { fg = fg,      bg = bg_l })
  hl(0, "FloatBorder",   { fg = cyan_d,  bg = bg_l })
  hl(0, "SignColumn",    { fg = fg_dim,  bg = bg })
  hl(0, "CursorLine",    { bg = bg_l })
  hl(0, "CursorColumn",  { bg = bg_l })
  hl(0, "CursorLineNr",  { fg = cyan,    bg = bg_l, bold = true })
  hl(0, "LineNr",        { fg = "#4b5263", bg = bg })
  hl(0, "Visual",        { bg = "#24314a" })
  hl(0, "VertSplit",     { fg = "#15182a", bg = bg })
  hl(0, "ColorColumn",   { bg = "#15182a" })
  hl(0, "StatusLine",    { fg = fg,      bg = bg_l })
  hl(0, "StatusLineNC",  { fg = fg_dim,  bg = bg_d })
  hl(0, "TabLine",       { fg = fg_dim,  bg = bg_d })
  hl(0, "TabLineSel",    { fg = cyan,    bg = bg_l, bold = true })
  hl(0, "TabLineFill",   { fg = fg_dim,  bg = bg_d })

  -- Popup / menu
  hl(0, "Pmenu",         { fg = fg,      bg = bg_l })
  hl(0, "PmenuSel",      { fg = bg,      bg = cyan })
  hl(0, "PmenuSbar",     { bg = bg_d })
  hl(0, "PmenuThumb",    { bg = cyan_d })

  -- Syntax
  hl(0, "Comment",       { fg = "#4b5263", italic = true })
  hl(0, "Constant",      { fg = yellow })
  hl(0, "String",        { fg = green })
  hl(0, "Character",     { fg = green })
  hl(0, "Number",        { fg = yellow })
  hl(0, "Boolean",       { fg = yellow })
  hl(0, "Identifier",    { fg = cyan })
  hl(0, "Function",      { fg = cyan, bold = true })
  hl(0, "Statement",     { fg = magenta })
  hl(0, "Conditional",   { fg = magenta })
  hl(0, "Repeat",        { fg = magenta })
  hl(0, "Operator",      { fg = fg })
  hl(0, "Keyword",       { fg = magenta, italic = true })
  hl(0, "Exception",     { fg = magenta })
  hl(0, "PreProc",       { fg = orange })
  hl(0, "Include",       { fg = orange })
  hl(0, "Define",        { fg = orange })
  hl(0, "Type",          { fg = cyan_d })
  hl(0, "StorageClass",  { fg = cyan_d })
  hl(0, "Structure",     { fg = cyan_d })
  hl(0, "Typedef",       { fg = cyan_d })
  hl(0, "Special",       { fg = yellow })
  hl(0, "Delimiter",     { fg = fg })
  hl(0, "Error",         { fg = bg, bg = red, bold = true })
  hl(0, "ErrorMsg",      { fg = bg, bg = red, bold = true })
  hl(0, "WarningMsg",    { fg = orange, bold = true })
  hl(0, "MoreMsg",       { fg = green, bold = true })
  hl(0, "Question",      { fg = green, bold = true })

  -- Diff
  hl(0, "DiffAdd",       { fg = green,  bg = "#0b2b19" })
  hl(0, "DiffChange",    { fg = yellow, bg = "#26210b" })
  hl(0, "DiffDelete",    { fg = red,    bg = "#2b0b12" })
  hl(0, "DiffText",      { fg = cyan,   bg = "#0b2230" })

  -- Statusline accents
  hl(0, "StatusLineAccent", { fg = bg,   bg = cyan,   bold = true })
  hl(0, "StatusLineMode",   { fg = bg,   bg = magenta, bold = true })

  -- Diagnostics (LSP)
  hl(0, "DiagnosticError", { fg = red })
  hl(0, "DiagnosticWarn",  { fg = yellow })
  hl(0, "DiagnosticInfo",  { fg = cyan })
  hl(0, "DiagnosticHint",  { fg = fg_dim })
end

set_neon_theme()

-----------------------------------------------------------
-- 4. Cursor Style (a bit more sci-fi)
-----------------------------------------------------------
vim.o.guicursor =
  "n-v-c:block-Cursor/lCursor," ..
  "i-ci:ver25-CursorInsert," ..
  "r-cr:hor20-CursorReplace," ..
  "o:hor50-Cursor," ..
  "a:blinkwait700-blinkon400-blinkoff250"

-----------------------------------------------------------
-- 5. Statusline (No plugins, just pure % codes)
-----------------------------------------------------------
local function set_statusline()
  -- mode names: make them a bit cooler
  local mode_map = {
    n = "NORMAL",
    no = "O-PENDING",
    v = "VISUAL",
    V = "V-LINE",
    [""] = "V-BLOCK",
    s = "SELECT",
    S = "S-LINE",
    [""] = "S-BLOCK",
    i = "INSERT",
    R = "REPLACE",
    c = "COMMAND",
    r = "PROMPT",
    t = "TERMINAL",
  }

  -- Expose a Lua function for dynamic mode in statusline
  _G.NeonMode = function()
    local m = vim.fn.mode()
    return mode_map[m] or m
  end

  vim.o.statusline = table.concat({
    "%#StatusLineAccent# ⦿ ",         -- left neon blob
    "%#StatusLine# %f ",              -- file path
    "%#StatusLineNC#%m%r%h%w",        -- modified/readonly flags
    "%=",                             -- right align from here
    "%#StatusLine# [%{&filetype!=''?&filetype:'no-ft'}] ",
    "%#StatusLineMode# %{v:lua.NeonMode()} ", -- mode
    "%#StatusLine#  ln %l/%L  col %c ",
  })
end

set_statusline()

-----------------------------------------------------------
-- 6. Tabline (simple but stylized)
-----------------------------------------------------------
_G.NeonTabline = function()
  local s = ""

  for i = 1, vim.fn.tabpagenr("$") do
    local winnr = vim.fn.tabpagewinnr(i)
    local bufnr = vim.fn.tabpagebuflist(i)[winnr]
    local name = vim.fn.fnamemodify(vim.fn.bufname(bufnr), ":t")
    if name == "" then name = "[No Name]" end

    if i == vim.fn.tabpagenr() then
      s = s .. "%#TabLineSel# " .. i .. ": " .. name .. " "
    else
      s = s .. "%#TabLine# " .. i .. ": " .. name .. " "
    end
  end

  s = s .. "%#TabLineFill#"
  return s
end

vim.o.tabline = "%!v:lua.NeonTabline()"

-----------------------------------------------------------
-- 7. Keymaps (Navigation, splits, quality-of-life)
-----------------------------------------------------------
local map = vim.keymap.set
local opts = { noremap = true, silent = true }

-- Better window navigation
map("n", "<C-h>", "<C-w>h", opts)
map("n", "<C-j>", "<C-w>j", opts)
map("n", "<C-k>", "<C-w>k", opts)
map("n", "<C-l>", "<C-w>l", opts)

-- Split windows
map("n", "<leader>v", ":vsplit<CR>", opts)
map("n", "<leader>h", ":split<CR>", opts)

-- Quick save & quit
map("n", "<leader>w", ":w<CR>", opts)
map("n", "<leader>q", ":q<CR>", opts)
map("n", "<leader>Q", ":qa!<CR>", opts)

-- Toggle relative number
map("n", "<leader>rn", function()
  vim.wo.relativenumber = not vim.wo.relativenumber
end, { silent = true, desc = "Toggle relative number" })

-- Clear search highlight
map("n", "<leader><space>", ":nohlsearch<CR>", opts)

-- Fast escape from insert
map("i", "jk", "<Esc>", opts)

-- Move lines up/down in visual mode
map("v", "J", ":m '>+1<CR>gv=gv", opts)
map("v", "K", ":m '<-2<CR>gv=gv", opts)

-- Open terminal in split
map("n", "<leader>tt", ":split | terminal<CR>", opts)

-----------------------------------------------------------
-- 8. Language-specific settings
-----------------------------------------------------------
-- Python: 4 spaces, no tabs
vim.api.nvim_create_autocmd("FileType", {
  pattern = { "python" },
  callback = function()
    vim.bo.shiftwidth = 4
    vim.bo.tabstop = 4
    vim.bo.expandtab = true
  end,
})

-- Bash / sh: 2 spaces (tweak if you prefer 4)
vim.api.nvim_create_autocmd("FileType", {
  pattern = { "sh", "bash", "zsh" },
  callback = function()
    vim.bo.shiftwidth = 2
    vim.bo.tabstop = 2
    vim.bo.expandtab = true
  end,
})

-- C / C++: 4 spaces
vim.api.nvim_create_autocmd("FileType", {
  pattern = { "c", "cpp", "h", "hpp" },
  callback = function()
    vim.bo.shiftwidth = 4
    vim.bo.tabstop = 4
    vim.bo.expandtab = true
  end,
})

-- Rust: 4 spaces, like rustfmt
vim.api.nvim_create_autocmd("FileType", {
  pattern = { "rust" },
  callback = function()
    vim.bo.shiftwidth = 4
    vim.bo.tabstop = 4
    vim.bo.expandtab = true
  end,
})

-- Go: tabs, width 4 (Go standard)
vim.api.nvim_create_autocmd("FileType", {
  pattern = { "go" },
  callback = function()
    vim.bo.shiftwidth = 4
    vim.bo.tabstop = 4
    vim.bo.expandtab = false
  end,
})

-----------------------------------------------------------
-- 9. Minimal, Safe LSP & Diagnostics (No external Lua deps)
--    NOTE: You still need language servers installed as binaries:
--    - Python: pyright-langserver
--    - C/C++: clangd
--    - Bash: bash-language-server
--    - Rust: rust-analyzer
--    - Go: gopls
-----------------------------------------------------------
if vim.lsp and vim.lsp.start then
  -- Diagnostics appearance
  vim.diagnostic.config({
    virtual_text = true,
    signs = true,
    underline = true,
    update_in_insert = false,
    severity_sort = true,
  })

  local signs = { Error = " ", Warn = " ", Hint = " ", Info = " " }
  for type, icon in pairs(signs) do
    local hl = "DiagnosticSign" .. type
    vim.fn.sign_define(hl, { text = icon, texthl = hl, numhl = "" })
  end

  local function on_attach(client, bufnr)
    local bufmap = function(mode, lhs, rhs)
      vim.keymap.set(mode, lhs, rhs, { noremap = true, silent = true, buffer = bufnr })
    end

    bufmap("n", "gd", vim.lsp.buf.definition)
    bufmap("n", "K", vim.lsp.buf.hover)
    bufmap("n", "gi", vim.lsp.buf.implementation)
    bufmap("n", "gr", vim.lsp.buf.references)
    bufmap("n", "<leader>rn", vim.lsp.buf.rename)
    bufmap("n", "<leader>ca", vim.lsp.buf.code_action)
    bufmap("n", "[d", vim.diagnostic.goto_prev)
    bufmap("n", "]d", vim.diagnostic.goto_next)
    bufmap("n", "<leader>f", function() vim.lsp.buf.format({ async = true }) end)
  end

  local function basic_lsp_setup(ft, cmd)
    vim.api.nvim_create_autocmd("FileType", {
      pattern = ft,
      callback = function(args)
        local bufnr = args.buf
        -- Start an LSP for this buffer if not already running for the same root
        vim.lsp.start({
          name = cmd[1],
          cmd = cmd,
          root_dir = vim.fn.getcwd(), -- simple but safe
          on_attach = on_attach,
        })
      end,
    })
  end

  -- Python
  basic_lsp_setup({ "python" }, { "pyright-langserver", "--stdio" })

  -- C / C++
  basic_lsp_setup({ "c", "cpp" }, { "clangd" })

  -- Bash
  basic_lsp_setup({ "sh", "bash" }, { "bash-language-server", "start" })

  -- Rust
  basic_lsp_setup({ "rust" }, { "rust-analyzer" })

  -- Go
  basic_lsp_setup({ "go" }, { "gopls" })
end

-----------------------------------------------------------
-- 10. Final touches
-----------------------------------------------------------
-- Slightly dim inactive windows
vim.api.nvim_create_autocmd({ "WinEnter", "BufWinEnter" }, {
  callback = function()
    vim.wo.cursorline = true
  end,
})
vim.api.nvim_create_autocmd("WinLeave", {
  callback = function()
    vim.wo.cursorline = false
  end,
})

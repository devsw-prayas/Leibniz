; asmAdoxU64.asm
; Implements ADOX (overflow-chain addition)

OPTION CASEMAP:NONE

.code
PUBLIC asmAdoxU64

asmAdoxU64 PROC
    ; RCX = pointer to a
    ; RDX = b

    mov rax, [rcx]     ; load *a
    adox rax, rdx      ; rax += b + OF
    mov [rcx], rax     ; store result

    ret
asmAdoxU64 ENDP

END
; asmAdcxU64.asm
; Implements ADCX (carry-chain addition)

OPTION CASEMAP:NONE

.code
PUBLIC asmAdcxU64

asmAdcxU64 PROC
    ; RCX = pointer to a
    ; RDX = b

    mov rax, [rcx]     ; load *a
    adcx rax, rdx      ; rax += b + CF
    mov [rcx], rax     ; store result

    ret
asmAdcxU64 ENDP

END
	.file	"1.cpp"
	.section	.rodata
.LC0:
	.string	"%ld"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$2064, %rsp
	movl	$0, -8(%rbp)
	jmp	.L2
.L7:
	movl	-8(%rbp), %eax
	cltq
	movq	%rax, -16(%rbp)
	movl	$0, -4(%rbp)
	jmp	.L3
.L6:
	movq	-16(%rbp), %rax
	andl	$1, %eax
	testb	%al, %al
	je	.L4
	movq	-16(%rbp), %rax
	movq	%rax, %rdx
	shrq	%rdx
	movl	$3988292384, %eax
	xorq	%rdx, %rax
	jmp	.L5
.L4:
	movq	-16(%rbp), %rax
	shrq	%rax
.L5:
	movq	%rax, -16(%rbp)
	addl	$1, -4(%rbp)
.L3:
	cmpl	$7, -4(%rbp)
	setle	%al
	testb	%al, %al
	jne	.L6
	movl	-8(%rbp), %eax
	cltq
	movq	-16(%rbp), %rdx
	movq	%rdx, -2064(%rbp,%rax,8)
	addl	$1, -8(%rbp)
.L2:
	cmpl	$255, -8(%rbp)
	setle	%al
	testb	%al, %al
	jne	.L7
	movq	-1984(%rbp), %rax
	movq	%rax, %rsi
	movl	$.LC0, %edi
	movl	$0, %eax
	call	printf
	movl	$0, %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.ident	"GCC: (Ubuntu/Linaro 4.6.1-9ubuntu3) 4.6.1"
	.section	.note.GNU-stack,"",@progbits

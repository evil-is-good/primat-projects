	.file	"1.go"
	.section	.go_export,"",@progbits
	.ascii	"v1;\n"
	.ascii	"package "
	.ascii	"main"
	.ascii	";\n"
	.ascii	"prefix "
	.ascii	"go"
	.ascii	";\n"
	.ascii	"priority 1;\n"
	.ascii	"func "
	.ascii	"main"
	.ascii	" ("
	.ascii	")"
	.ascii	";\n"
	.ascii	"checksum 1513599C504AB98D30795C7D32881B053E28650B;\n"
	.text
	.p2align 4,,15
	.globl	main.main
	.type	main.main, @function
main.main:
.LFB0:
	.cfi_startproc
	cmpq	%fs:112, %rsp
	jb	.L4
	ret
.L4:
	xorl	%r10d, %r10d
	xorl	%r11d, %r11d
	call	__morestack
	ret
	ret
	.cfi_endproc
.LFE0:
	.size	main.main, .-main.main
	.p2align 4,,15
	.globl	__go_init_main
	.type	__go_init_main, @function
__go_init_main:
.LFB1:
	.cfi_startproc
	cmpq	%fs:112, %rsp
	jb	.L7
	ret
.L7:
	xorl	%r10d, %r10d
	xorl	%r11d, %r11d
	call	__morestack
	ret
	ret
	.cfi_endproc
.LFE1:
	.size	__go_init_main, .-__go_init_main
	.section	.note.GNU-split-stack,"",@progbits
	.ident	"GCC: (Ubuntu/Linaro 4.6.1-9ubuntu3) 4.6.1"
	.section	.note.GNU-stack,"",@progbits

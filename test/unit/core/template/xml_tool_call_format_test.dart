import 'dart:convert';

import 'package:llamadart/src/core/template/xml_tool_call_format.dart';
import 'package:test/test.dart';

void main() {
  test('parseXmlToolCalls parses a simple XML-like tool call', () {
    const output =
        '<tool_call>\n'
        '<function=weather>\n'
        '<parameter=city>\n"Seoul"\n</parameter>\n'
        '</function>\n'
        '</tool_call>';
    final parsed = parseXmlToolCalls(output, XmlToolCallFormat.qwen3Coder);

    expect(parsed.toolCalls, hasLength(1));
    expect(parsed.toolCalls.first.function?.name, 'weather');
    expect(jsonDecode(parsed.toolCalls.first.function!.arguments!), {
      'city': 'Seoul',
    });
  });

  test('parseXmlToolCalls preserves raw argument strings when not JSON', () {
    const output =
        '<tool_call>\n'
        '<function=weather>\n'
        '<parameter=city>\nSeoul\n</parameter>\n'
        '</function>\n'
        '</tool_call>';
    final parsed = parseXmlToolCalls(output, XmlToolCallFormat.qwen3Coder);

    expect(parsed.toolCalls, hasLength(1));
    expect(parsed.toolCalls.first.function?.name, 'weather');
    expect(jsonDecode(parsed.toolCalls.first.function!.arguments!), {
      'city': 'Seoul',
    });
  });

  test('parseXmlToolCalls keeps malformed scoped XML payload as content', () {
    const output =
        '<tool_call>\n'
        '<function=weather>\n'
        '<parameter=city>\n"Seoul"\n'
        '</tool_call>';
    final parsed = parseXmlToolCalls(output, XmlToolCallFormat.qwen3Coder);

    expect(parsed.toolCalls, isEmpty);
    expect(parsed.content, output);
  });

  test('parseXmlToolCalls keeps malformed scoped prelude as content', () {
    const output =
        '<tool_call>\n'
        'oops\n'
        '<function=weather>\n'
        '<parameter=city>\n"Seoul"\n</parameter>\n'
        '</function>\n'
        '</tool_call>';
    final parsed = parseXmlToolCalls(output, XmlToolCallFormat.qwen3Coder);

    expect(parsed.toolCalls, isEmpty);
    expect(parsed.content, output);
  });
}
